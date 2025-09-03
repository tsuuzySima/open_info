"""
Open3D + NumPy による 非反復走査LiDAR向け 背景差分侵入検知（疎グリッド）
- 球座標グリッド（r, theta, phi）で短期窓(ST)の観測カウントと長期窓(LB)の期待カウント(λ)を管理
- Poisson正規化zスコア + 自由空間違反で候補抽出
- Open3DのDBSCANで3Dクラスタ化
- 簡易Kalmanフィルタで追跡（CVモデル, 3D）
- 背景更新は侵入クラスタ近傍を凍結

メモ:
- theta: 水平方位角 [-π, π], phi: 垂直仰角 [-π/2, π/2]
- FOVはパラメータで制限、格子は疎（dict）で保持
- 実センサではIMU/時間同期/外乱補正を追加推奨
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import numpy as np
import open3d as o3d
import math
import time

# ----------------------------
# 1) パラメータ定義
# ----------------------------

@dataclass
class GridParams:
    # 距離
    r_max: float = 250.0
    dr: float = 0.5  # 0.25–0.5mを推奨（遠距離優先なら0.5m）
    # 角度（FOV）。LiDAR機種・実設置に合わせ調整
    theta_min_deg: float = -60.0
    theta_max_deg: float =  60.0
    dtheta_deg: float   =   0.1
    phi_min_deg:   float = -40.0
    phi_max_deg:   float =  40.0
    dphi_deg:     float  =   0.1

    def __post_init__(self):
        self.theta_min = math.radians(self.theta_min_deg)
        self.theta_max = math.radians(self.theta_max_deg)
        self.dtheta    = math.radians(self.dtheta_deg)
        self.phi_min   = math.radians(self.phi_min_deg)
        self.phi_max   = math.radians(self.phi_max_deg)
        self.dphi      = math.radians(self.dphi_deg)

        # ビン数
        self.n_r = int(self.r_max / self.dr)
        self.n_theta = max(1, int((self.theta_max - self.theta_min) / self.dtheta))
        self.n_phi   = max(1, int((self.phi_max - self.phi_min) / self.dphi))


@dataclass
class ShortTermParams:
    # 短期積分窓（秒）。0.3–1.0s目安
    window_sec: float = 0.5
    # カバレッジ判定：ST内でその角度ビンに「最低何回ヒットがあれば観測済」とみなすか
    min_hits_angle: int = 1


@dataclass
class LongTermParams:
    # 長期窓の半減期（秒）：静的環境の学習速度
    half_life_sec: float = 45.0
    # 期待カウント(λ)の最小・最大でのクリップ（数値安定性）
    lambda_min: float = 0.0
    lambda_max: float = 50.0
    # 自由空間（最近接背景面）推定の半減期
    free_half_life_sec: float = 60.0
    # 自由空間違反の許容マージン [m]
    free_margin_m: float = 1.0
    # λ辞書の掃除閾値（この値未満になった古いセルは破棄）
    lambda_prune_th: float = 0.02


@dataclass
class DetectionParams:
    # Poisson zスコアしきい値
    z_th: float = 4.0
    # クラスタリング
    dbscan_eps_m: float = 0.8     # 0.5–1.0m程度（遠距離は0.8–1.2m）
    dbscan_min_points: int = 6
    # クラスタのサイズ妥当性（AABBの各辺[m]）
    min_size_m: float = 0.3
    max_size_m: float = 3.0
    # 自由空間違反比率（クラスタ内の点のうち何%が違反か）
    free_violation_ratio_th: float = 0.4
    # 距離依存の最小点数（200mで6–10点目安）: base + slope * (R/100m)
    min_points_base: int = 6
    min_points_slope_per100m: float = 4.0  # 100mあたり加算


@dataclass
class TrackingParams:
    # 追跡（CV, 3D）
    dt_default: float = 0.5  # ST窓に合わせる
    process_noise: float = 1.0   # Qのスケール
    meas_noise: float = 0.5      # 測定ノイズ（位置[m]）
    gate_dist_m: float = 3.0     # データ関連付けゲート
    # M-of-N確証ロジック
    confirm_N: int = 5
    confirm_M: int = 3
    # ロスト許容
    max_missed: int = 4


# ----------------------------
# 2) ユーティリティ
# ----------------------------

def cart_to_sph(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xyz -> (r, theta(azimuth), phi(elevation))"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(y, x)  # [-π, π]
    xy = np.sqrt(x * x + y * y)
    phi = np.arctan2(z, xy)   # [-π/2, π/2]
    return r, theta, phi


def discretize(vals: np.ndarray, vmin: float, dv: float, n_bins: int) -> np.ndarray:
    """連続値 -> ビンインデックス（範囲外は-1）"""
    idx = np.floor((vals - vmin) / dv).astype(np.int32)
    idx[(idx < 0) | (idx >= n_bins)] = -1
    return idx


def pack_key(ir: int, it: int, ip: int, n_theta: int, n_phi: int) -> int:
    """(ir, it, ip) -> 64bitキー"""
    if ir < 0 or it < 0 or ip < 0:
        return -1
    return (ir * n_theta + it) * n_phi + ip


def exp_moving_update(old: float, new: float, alpha: float) -> float:
    """指数移動平均：old*(1-alpha) + new*alpha"""
    return old * (1.0 - alpha) + new * alpha


def half_life_to_alpha(dt: float, half_life: float) -> float:
    """半減期→指数移動平均のα（dt更新ごと）"""
    if half_life <= 0:
        return 1.0
    # old * 0.5^(dt/half_life) + new * (1 - 0.5^(dt/half_life))
    return 1.0 - math.pow(0.5, dt / half_life)


# ----------------------------
# 3) 短期窓(ST)と長期窓(LB)のモデル
# ----------------------------

class ShortTermWindow:
    """短期窓（0.3–1.0s）内の観測統計"""
    def __init__(self, grid: GridParams, stp: ShortTermParams):
        self.g = grid
        self.p = stp
        # 疎カウント辞書: key -> k (int)
        self.counts: Dict[int, int] = {}
        # 角度ビン観測（カバレッジ）
        self.angle_hits = np.zeros((self.g.n_theta, self.g.n_phi), dtype=np.int32)
        # 角度ごとの最近接距離（free仮定用）
        self.min_r_angle = np.full((self.g.n_theta, self.g.n_phi), np.inf, dtype=np.float32)

    def integrate_points(self, pts_xyz: np.ndarray):
        if pts_xyz.size == 0:
            return
        r, theta, phi = cart_to_sph(pts_xyz)
        ir = discretize(r, 0.0, self.g.dr, self.g.n_r)
        it = discretize(theta, self.g.theta_min, self.g.dtheta, self.g.n_theta)
        ip = discretize(phi,   self.g.phi_min,   self.g.dphi,   self.g.n_phi)

        for i in range(pts_xyz.shape[0]):
            if ir[i] < 0 or it[i] < 0 or ip[i] < 0:
                continue
            key = pack_key(ir[i], it[i], ip[i], self.g.n_theta, self.g.n_phi)
            self.counts[key] = self.counts.get(key, 0) + 1
            self.angle_hits[it[i], ip[i]] += 1
            if r[i] < self.min_r_angle[it[i], ip[i]]:
                self.min_r_angle[it[i], ip[i]] = r[i]

    def is_angle_covered(self, it: int, ip: int) -> bool:
        return self.angle_hits[it, ip] >= self.p.min_hits_angle


class LongTermBackground:
    """長期背景モデル（λと自由空間距離）"""
    def __init__(self, grid: GridParams, lbp: LongTermParams, dt_nominal: float):
        self.g = grid
        self.p = lbp
        self.dt = dt_nominal
        # λ: key -> (value, last_step). lazy decayで管理
        self.lam: Dict[int, Tuple[float, int]] = {}
        self.step = 0
        # 自由空間：角度ごとの最近接背景距離（EMAで推定）
        self.free_r_bg = np.full((self.g.n_theta, self.g.n_phi), np.inf, dtype=np.float32)
        # 初期値：未知なら∞（自由とする）。実環境で初期ウォームアップ推奨。

    def _decay_lambda(self, value: float, last_step: int) -> float:
        """lazy decay: 現在stepまで指数減衰"""
        steps = self.step - last_step
        if steps <= 0:
            return value
        alpha = half_life_to_alpha(self.dt * steps, self.p.half_life_sec)
        # decayのみだと new = old*(1-alpha) + 0*alpha
        return value * (1.0 - alpha)

    def get_lambda(self, key: int) -> float:
        if key not in self.lam:
            return 0.0
        v, last = self.lam[key]
        return max(self.p.lambda_min, min(self.p.lambda_max, self._decay_lambda(v, last)))

    def set_lambda(self, key: int, new_value: float):
        self.lam[key] = (max(self.p.lambda_min, min(self.p.lambda_max, new_value)), self.step)

    def step_update(self, st: ShortTermWindow, freeze_keys: Optional[set] = None):
        """ST統計を背景に取り込み（凍結セルはスキップ）"""
        self.step += 1
        if freeze_keys is None:
            freeze_keys = set()

        alpha_lam = half_life_to_alpha(self.dt, self.p.half_life_sec)
        # 1) まず既存λを自然減衰（lazyではアクセス時に減衰するが、掃除のため一部実施）
        # ここでは軽量化のため、辞書サイズが大きい場合はスキップ可
        keys_to_delete = []
        for key, (v, last) in self.lam.items():
            v_dec = self._decay_lambda(v, last)
            if v_dec < self.p.lambda_prune_th:
                keys_to_delete.append(key)
            else:
                # last_stepを現在step-1に戻しておく（次回のdecay差分を小さく）
                self.lam[key] = (v_dec, self.step - 1)
        for k in keys_to_delete:
            del self.lam[k]

        # 2) ST観測をEMAで反映（凍結セル除外）
        for key, k_cnt in st.counts.items():
            if key in freeze_keys:
                continue
            prev = self.get_lambda(key)
            newv = exp_moving_update(prev, float(k_cnt), alpha_lam)
            self.set_lambda(key, newv)

        # 3) 自由空間（最近接背景面）の更新
        alpha_free = half_life_to_alpha(self.dt, self.p.free_half_life_sec)
        # ST内で観測された角度についてmin_rをEMA
        observed_mask = np.isfinite(st.min_r_angle)
        # unknown(=inf)は更新しない
        it_idx, ip_idx = np.where(observed_mask)
        for it, ip in zip(it_idx, ip_idx):
            st_min_r = st.min_r_angle[it, ip]
            cur = self.free_r_bg[it, ip]
            if math.isfinite(st_min_r):
                if not math.isfinite(cur):
                    self.free_r_bg[it, ip] = st_min_r
                else:
                    self.free_r_bg[it, ip] = exp_moving_update(cur, st_min_r, alpha_free)


# ----------------------------
# 4) 検知器 + 追跡
# ----------------------------

@dataclass
class ClusterDetection:
    points: np.ndarray                # (N,3)
    centroid: np.ndarray              # (3,)
    mean_range: float
    free_violation_ratio: float
    aabb_size: np.ndarray             # (3,) 各軸長
    score: float                      # 任意スコア（z等の統計を入れてもよい）


class KalmanTrack:
    """定速度(CV) 3D Kalmanトラック x=[x,y,z,vx,vy,vz]"""
    _next_id = 1

    def __init__(self, z_pos: np.ndarray, tp: TrackingParams):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.tp = tp

        self.x = np.zeros(6)
        self.x[0:3] = z_pos
        self.P = np.eye(6) * 3.0  # 初期共分散

        self.missed = 0
        self.hits_history: List[bool] = []

    def predict(self, dt: float):
        if dt <= 0:
            dt = self.tp.dt_default
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        Q = np.eye(6) * self.tp.process_noise
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pos: np.ndarray):
        H = np.zeros((3,6))
        H[0,0] = H[1,1] = H[2,2] = 1.0
        R = np.eye(3) * (self.tp.meas_noise ** 2)
        y = z_pos - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.missed = 0
        self.hits_history.append(True)
        if len(self.hits_history) > self.tp.confirm_N:
            self.hits_history = self.hits_history[-self.tp.confirm_N:]

    def miss(self):
        self.missed += 1
        self.hits_history.append(False)
        if len(self.hits_history) > self.tp.confirm_N:
            self.hits_history = self.hits_history[-self.tp.confirm_N:]

    def is_confirmed(self) -> bool:
        # 直近N中M以上のヒット
        if len(self.hits_history) < self.tp.confirm_N:
            return False
        return sum(self.hits_history[-self.tp.confirm_N:]) >= self.tp.confirm_M

    def position(self) -> np.ndarray:
        return self.x[0:3]

    def velocity(self) -> np.ndarray:
        return self.x[3:6]


class Detector:
    def __init__(self,
                 grid: GridParams = GridParams(),
                 stp: ShortTermParams = ShortTermParams(),
                 lbp: LongTermParams = LongTermParams(),
                 detp: DetectionParams = DetectionParams(),
                 trkp: TrackingParams = TrackingParams()):
        self.g = grid
        self.stp = stp
        self.lbp = lbp
        self.detp = detp
        self.trkp = trkp

        self.lb = LongTermBackground(self.g, self.lbp, dt_nominal=self.stp.window_sec)
        self.tracks: List[KalmanTrack] = []
        self.last_time: Optional[float] = None

    # ---- 内部: 候補抽出 ----
    def _extract_candidates(self, pts_xyz: np.ndarray, st: ShortTermWindow) -> Tuple[np.ndarray, set]:
        """候補点（変化セル or 自由空間違反セル）と、凍結セル集合を返す"""
        if pts_xyz.size == 0:
            return np.empty((0,3)), set()

        # 1) 座標->ビン
        r, theta, phi = cart_to_sph(pts_xyz)
        ir = discretize(r, 0.0, self.g.dr, self.g.n_r)
        it = discretize(theta, self.g.theta_min, self.g.dtheta, self.g.n_theta)
        ip = discretize(phi,   self.g.phi_min,   self.g.dphi,   self.g.n_phi)

        candidate_keys = set()
        violation_mask = np.zeros(pts_xyz.shape[0], dtype=bool)

        # 2) Poisson正規化：STで観測したセルのみ評価
        for key, k_cnt in st.counts.items():
            # 角度カバレッジ（STで当該角度が十分観測されたか）を確認
            # key -> (ir,it,ip)へ逆変換
            tmp = key
            ip_ = tmp % self.g.n_phi; tmp //= self.g.n_phi
            it_ = tmp % self.g.n_theta; tmp //= self.g.n_theta
            # ir_ = tmp % self.g.n_r  # 使わないが参考
            if not st.is_angle_covered(it_, ip_):
                continue

            lam = self.lb.get_lambda(key)
            z = (k_cnt - lam) / math.sqrt(max(lam, 1.0))
            if z > self.detp.z_th:
                candidate_keys.add(key)

        # 3) 自由空間違反： r < free_r_bg(it,ip) - margin
        margin = self.lbp.free_margin_m
        # 角度ビンが無効な点は除外
        valid = (ir >= 0) & (it >= 0) & (ip >= 0)
        idxs = np.where(valid)[0]
        for i in idxs:
            if not st.is_angle_covered(it[i], ip[i]):
                continue
            bg_r = self.lb.free_r_bg[it[i], ip[i]]
            if math.isfinite(bg_r) and (r[i] + 1e-6) < (bg_r - margin):
                violation_mask[i] = True
                key = pack_key(ir[i], it[i], ip[i], self.g.n_theta, self.g.n_phi)
                candidate_keys.add(key)

        # 4) 候補点抽出
        #   - zスコアで変化セルになった点
        #   - もしくは自由空間違反の点
        cand_mask = violation_mask.copy()
        if len(candidate_keys) > 0:
            # セル候補に属する点を追加
            keys_arr = np.array(list(candidate_keys), dtype=np.int64)
            # 点ごとのkeyを計算
            keys_points = (ir * self.g.n_theta + it) * self.g.n_phi + ip
            in_set = np.isin(keys_points, keys_arr)
            cand_mask |= in_set

        candidate_points = pts_xyz[cand_mask]
        # 背景凍結：候補セル+周辺（ここでは簡略に候補セルのみ）
        freeze_keys = set(candidate_keys)
        return candidate_points, freeze_keys

    # ---- 内部: クラスタリング & 妥当性判定 ----
    def _cluster_and_filter(self, candidate_points: np.ndarray,
                            all_points: np.ndarray) -> List[ClusterDetection]:
        if candidate_points.shape[0] < self.detp.dbscan_min_points:
            return []

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidate_points))
        labels = np.array(pcd.cluster_dbscan(eps=self.detp.dbscan_eps_m,
                                             min_points=self.detp.dbscan_min_points,
                                             print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            return []

        dets: List[ClusterDetection] = []
        for cid in range(labels.max() + 1):
            idx = np.where(labels == cid)[0]
            pts = candidate_points[idx]
            if pts.shape[0] < self.detp.dbscan_min_points:
                continue

            # AABBサイズチェック
            aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(pts))
            extent = np.asarray(aabb.get_extent())  # (dx, dy, dz)
            if np.any(extent < self.detp.min_size_m) or np.any(extent > self.detp.max_size_m):
                # 遠距離では多少緩めたい場合はここを距離依存にしても良い
                pass  # サイズは厳しすぎる場合があるので通過させ、次段で点数条件で調整
            centroid = np.mean(pts, axis=0)
            rng = float(np.linalg.norm(centroid))

            # 距離依存の最小点数
            min_pts_range = int(self.detp.min_points_base +
                                self.detp.min_points_slope_per100m * (rng / 100.0))
            if pts.shape[0] < min_pts_range:
                continue

            # 自由空間違反比率（クラスタ内の点のうち何割が違反点だったか）
            # 簡略: 近傍点数を使わず、背景の最近接距離 free_r_bg と比較
            r, theta, phi = cart_to_sph(pts)
            it = discretize(theta, self.g.theta_min, self.g.dtheta, self.g.n_theta)
            ip = discretize(phi,   self.g.phi_min,   self.g.dphi,   self.g.n_phi)
            ok = (it >= 0) & (ip >= 0)
            vio = 0
            for i in range(pts.shape[0]):
                if not ok[i]:
                    continue
                bg_r = self.lb.free_r_bg[it[i], ip[i]]
                if math.isfinite(bg_r) and (r[i] + 1e-6) < (bg_r - self.lbp.free_margin_m):
                    vio += 1
            vio_ratio = (vio / float(pts.shape[0])) if pts.shape[0] > 0 else 0.0
            if vio_ratio < self.detp.free_violation_ratio_th:
                # Poisson差分だけで出たクラスタを取りこぼしたくなければ、この条件を弱める
                pass  # 場合によっては通す
            # スコア（簡易）：点数 * 違反比率
            score = pts.shape[0] * max(vio_ratio, 0.1)

            dets.append(ClusterDetection(points=pts,
                                         centroid=centroid,
                                         mean_range=rng,
                                         free_violation_ratio=vio_ratio,
                                         aabb_size=extent,
                                         score=score))
        return dets

    # ---- 内部: 追跡（Greedy関連付け） ----
    def _update_tracks(self, detections: List[ClusterDetection], now: float):
        if self.last_time is None:
            dt = self.trkp.dt_default
        else:
            dt = max(1e-3, now - self.last_time)
        self.last_time = now

        # 1) 予測
        for tr in self.tracks:
            tr.predict(dt)

        # 2) 関連付け（貪欲）：距離ゲート内で最も近いもの
        unmatched_dets = set(range(len(detections)))
        unmatched_trks = set(range(len(self.tracks)))
        assignments: List[Tuple[int,int]] = []

        if len(self.tracks) > 0 and len(detections) > 0:
            dist = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, tr in enumerate(self.tracks):
                tr_pos = tr.position()
                for j, det in enumerate(detections):
                    d = np.linalg.norm(det.centroid - tr_pos)
                    dist[i, j] = d
            # 最小距離順に割当
            used_tr = set()
            used_det = set()
            flat = [(dist[i, j], i, j) for i in range(dist.shape[0]) for j in range(dist.shape[1])]
            flat.sort(key=lambda x: x[0])
            for d, i, j in flat:
                if d > self.trkp.gate_dist_m:
                    break
                if i in used_tr or j in used_det:
                    continue
                assignments.append((i, j))
                used_tr.add(i)
                used_det.add(j)
            unmatched_trks = set(range(len(self.tracks))) - set(i for i, _ in assignments)
            unmatched_dets = set(range(len(detections))) - set(j for _, j in assignments)

        # 3) 更新
        for i, j in assignments:
            self.tracks[i].update(detections[j].centroid)
        for i in unmatched_trks:
            self.tracks[i].miss()
        # 4) 新規トラック生成
        for j in unmatched_dets:
            self.tracks.append(KalmanTrack(detections[j].centroid, self.trkp))

        # 5) クリーニング
        self.tracks = [t for t in self.tracks if t.missed <= self.trkp.max_missed]

    # ---- パブリックAPI ----
    def process_frame(self, pts_xyz: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        """
        1フレーム（またはST窓で積分した点群）を処理する。
        pts_xyz: (N,3) in meters, センサー座標系。IMU等で動作補正済みが望ましい。
        """
        if timestamp is None:
            timestamp = time.time()

        st = ShortTermWindow(self.g, self.stp)
        st.integrate_points(pts_xyz)

        # 変化候補抽出
        candidate_points, freeze_keys = self._extract_candidates(pts_xyz, st)

        # クラスタリング & フィルタ
        detections = self._cluster_and_filter(candidate_points, pts_xyz)

        # 追跡更新
        self._update_tracks(detections, now=timestamp)

        # 背景更新（候補周辺は凍結）
        self.lb.step_update(st, freeze_keys=freeze_keys)

        # 出力まとめ
        out_tracks = []
        for tr in self.tracks:
            out_tracks.append({
                "id": tr.id,
                "pos": tr.position().tolist(),
                "vel": tr.velocity().tolist(),
                "confirmed": tr.is_confirmed(),
                "missed": tr.missed,
            })
        out_dets = []
        for det in detections:
            out_dets.append({
                "centroid": det.centroid.tolist(),
                "range_m": det.mean_range,
                "free_violation_ratio": det.free_violation_ratio,
                "aabb_size": det.aabb_size.tolist(),
                "score": det.score,
                "num_points": det.points.shape[0],
            })

        return {
            "num_points": int(pts_xyz.shape[0]),
            "num_candidates": int(candidate_points.shape[0]),
            "num_detections": len(detections),
            "detections": out_dets,
            "tracks": out_tracks,
        }


# ----------------------------
# 5) デモ（PCD読み込み）
# ----------------------------

def load_pcd(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


if __name__ == "__main__":
    # 例: 連番PCDを処理（ディレクトリを適宜変更）
    import glob, os
    pcd_dir = "./pcd_samples"  # あなたのPCDフォルダに合わせて修正
    files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))

    det = Detector(
        grid=GridParams(
            r_max=250.0, dr=0.5,
            theta_min_deg=-60.0, theta_max_deg=60.0, dtheta_deg=0.1,
            phi_min_deg=-40.0, phi_max_deg=40.0, dphi_deg=0.1
        ),
        stp=ShortTermParams(window_sec=0.5, min_hits_angle=1),
        lbp=LongTermParams(half_life_sec=45.0, free_half_life_sec=60.0, free_margin_m=1.0),
        detp=DetectionParams(z_th=4.0, dbscan_eps_m=0.8, dbscan_min_points=6,
                             min_size_m=0.3, max_size_m=3.0,
                             free_violation_ratio_th=0.4,
                             min_points_base=6, min_points_slope_per100m=4.0),
        trkp=TrackingParams(dt_default=0.5, process_noise=1.0, meas_noise=0.5,
                            gate_dist_m=3.0, confirm_N=5, confirm_M=3, max_missed=4)
    )

    for f in files:
        pts = load_pcd(f)
        result = det.process_frame(pts)
        print(os.path.basename(f), result)

