// 逐像素求解(GPU 版),与 tone.rs 的 CPU 逻辑一一对应:
//   平滑色 → 按颜色相似度取邻域参考权重 w̄ → 已一致则直接用 → 带一致性约束的 NNLS → 换算网点
// mode 0:整块像素,输出各色版网点值(每个 u32 打包 4 个版)
// mode 1:抽样像素(x、y 都是 step 的倍数),输出 18 个权重,供 CPU 汇总成网格平均

struct Params {
    w: u32, h: u32, y0: u32, rows: u32,
    f: u32, lw: u32, lh: u32, npts: u32,
    base: u32, n_inks: u32, max_layers: u32, mode: u32,
    step: u32, min_dot: u32, halo_y0: u32, out_words: u32,
    mu: f32, sigma: f32, sum_w: f32, pad0: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> px: array<u32>;
@group(0) @binding(3) var<storage, read> uidx: array<u32>;
@group(0) @binding(4) var<storage, read> mixes: array<f32>;
@group(0) @binding(5) var<storage, read> guide: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> low: array<f32>;
@group(0) @binding(7) var<storage, read_write> outb: array<u32>;
// 每行完成计数:主机端据此确认这一块确实全部算完(防止显卡超时被中止却读到旧数据)
@group(0) @binding(8) var<storage, read_write> done: array<atomic<u32>>;

const MAXP: u32 = 18u;
const TOL: f32 = 1e-7;

var<private> G: array<array<f32, 18>, 18>;
var<private> C: array<f32, 18>;
var<private> X: array<f32, 18>;
var<private> Z: array<f32, 18>;
var<private> PAS: array<u32, 18>;
var<private> A: array<array<f32, 18>, 18>;
var<private> B: array<f32, 18>;
var<private> SOL: array<f32, 18>;
var<private> IDX: array<u32, 18>;
var<private> PIDX: array<u32, 18>;
var<private> WBAR: array<f32, 18>;
var<private> M: array<f32, 18>;
var<private> M0: array<f32, 18>;

fn unpack(u: u32) -> vec3<f32> {
    return vec3<f32>(f32((u >> 16u) & 255u), f32((u >> 8u) & 255u), f32(u & 255u)) / 255.0;
}

fn to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn lab_f(t: f32) -> f32 {
    if (t > 0.008856) {
        return pow(t, 1.0 / 3.0);
    }
    return 7.787 * t + 16.0 / 116.0;
}

fn rgb_to_lab(c: vec3<f32>) -> vec3<f32> {
    let r = to_linear(c.x);
    let g = to_linear(c.y);
    let b = to_linear(c.z);
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
    let fx = lab_f(x);
    let fy = lab_f(y);
    let fz = lab_f(z);
    return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

fn px_at(x: u32, y: u32) -> vec3<f32> {
    return unpack(px[(y - P.halo_y0) * P.w + x]);
}

// 3×3 平均后的 Lab(忽略扫描颗粒)
fn smooth_lab(x: u32, y: u32) -> vec3<f32> {
    var acc = vec3<f32>(0.0);
    var cnt = 0.0;
    let ya = select(y - 1u, 0u, y == 0u);
    let yb = min(y + 1u, P.h - 1u);
    let xa = select(x - 1u, 0u, x == 0u);
    let xb = min(x + 1u, P.w - 1u);
    for (var yy = ya; yy <= yb; yy++) {
        for (var xx = xa; xx <= xb; xx++) {
            acc += px_at(xx, yy);
            cnt += 1.0;
        }
    }
    return rgb_to_lab(acc / cnt);
}

// 周围 4 个网格:双线性 × 颜色相似度 加权的参考权重 → WBAR
fn sample_prior(x: u32, y: u32, lab: vec3<f32>) {
    let fx = clamp((f32(x) + 0.5) / f32(P.f) - 0.5, 0.0, f32(P.lw - 1u));
    let fy = clamp((f32(y) + 0.5) / f32(P.f) - 0.5, 0.0, f32(P.lh - 1u));
    let x0 = u32(fx);
    let y0 = u32(fy);
    let x1 = min(x0 + 1u, P.lw - 1u);
    let y1 = min(y0 + 1u, P.lh - 1u);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    var cells = array<u32, 4>(y0 * P.lw + x0, y0 * P.lw + x1, y1 * P.lw + x0, y1 * P.lw + x1);
    var bws = array<f32, 4>((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty);
    for (var p = 0u; p < MAXP; p++) {
        WBAR[p] = 0.0;
    }
    var total = 0.0;
    let s2 = 2.0 * P.sigma * P.sigma;
    for (var i = 0u; i < 4u; i++) {
        let c = cells[i];
        let d = lab - guide[c].xyz;
        let wgt = bws[i] * exp(-dot(d, d) / s2) + 1e-6 * bws[i];
        for (var p = 0u; p < MAXP; p++) {
            WBAR[p] += low[c * MAXP + p] * wgt;
        }
        total += wgt;
    }
    for (var p = 0u; p < MAXP; p++) {
        WBAR[p] /= max(total, 1e-12);
    }
}

// 解 A·SOL = B(k×k,高斯消元 + 部分主元)
fn solve_linear(k: u32) {
    for (var i = 0u; i < k; i++) {
        A[i][i] += 1e-6;
    }
    for (var col = 0u; col < k; col++) {
        var piv = col;
        for (var r = col + 1u; r < k; r++) {
            if (abs(A[r][col]) > abs(A[piv][col])) {
                piv = r;
            }
        }
        if (piv != col) {
            for (var c = 0u; c < k; c++) {
                let t = A[col][c];
                A[col][c] = A[piv][c];
                A[piv][c] = t;
            }
            let tb = B[col];
            B[col] = B[piv];
            B[piv] = tb;
        }
        let d = A[col][col];
        if (abs(d) < 1e-12) {
            continue;
        }
        for (var r = col + 1u; r < k; r++) {
            let fct = A[r][col] / d;
            if (fct != 0.0) {
                for (var c = col; c < k; c++) {
                    A[r][c] -= fct * A[col][c];
                }
                B[r] -= fct * B[col];
            }
        }
    }
    for (var ii = 0u; ii < k; ii++) {
        let i = k - 1u - ii;
        var s = B[i];
        for (var j = i + 1u; j < k; j++) {
            s -= A[i][j] * SOL[j];
        }
        if (abs(A[i][i]) < 1e-12) {
            SOL[i] = 0.0;
        } else {
            SOL[i] = s / A[i][i];
        }
    }
}

// 在当前被动集上解最小二乘 → Z
fn ls_passive(n: u32) {
    var k = 0u;
    for (var j = 0u; j < n; j++) {
        if (PAS[j] == 1u) {
            IDX[k] = j;
            k++;
        }
    }
    for (var r = 0u; r < k; r++) {
        for (var q = 0u; q < k; q++) {
            A[r][q] = G[IDX[r]][IDX[q]];
        }
        B[r] = C[IDX[r]];
    }
    solve_linear(k);
    for (var j = 0u; j < n; j++) {
        Z[j] = 0.0;
    }
    for (var r = 0u; r < k; r++) {
        Z[IDX[r]] = SOL[r];
    }
}

// Lawson–Hanson NNLS(格拉姆形式)→ X
fn nnls(n: u32) {
    for (var j = 0u; j < n; j++) {
        X[j] = 0.0;
        PAS[j] = 0u;
    }
    for (var it = 0u; it < 3u * n; it++) {
        var best = n;
        var best_g = TOL;
        for (var j = 0u; j < n; j++) {
            if (PAS[j] == 0u) {
                var g = C[j];
                for (var i = 0u; i < n; i++) {
                    g -= G[j][i] * X[i];
                }
                if (g > best_g) {
                    best = j;
                    best_g = g;
                }
            }
        }
        if (best == n) {
            break;
        }
        PAS[best] = 1u;
        for (var inner = 0u; inner < 3u * n; inner++) {
            ls_passive(n);
            var allpos = true;
            for (var j = 0u; j < n; j++) {
                if (PAS[j] == 1u && Z[j] <= TOL) {
                    allpos = false;
                }
            }
            if (allpos) {
                for (var j = 0u; j < n; j++) {
                    X[j] = select(0.0, Z[j], PAS[j] == 1u);
                }
                break;
            }
            var alpha = 3.4e38;
            for (var j = 0u; j < n; j++) {
                if (PAS[j] == 1u && Z[j] <= TOL) {
                    alpha = min(alpha, X[j] / (X[j] - Z[j]));
                }
            }
            var any = false;
            for (var j = 0u; j < n; j++) {
                if (PAS[j] == 1u) {
                    X[j] += alpha * (Z[j] - X[j]);
                    if (X[j] <= TOL) {
                        PAS[j] = 0u;
                        X[j] = 0.0;
                    } else {
                        any = true;
                    }
                }
            }
            if (!any) {
                break;
            }
        }
    }
}

// 带一致性约束的混合求解 → M
fn solve_reg(t: vec3<f32>, allowed_in: u32) {
    let s2 = P.sum_w * P.sum_w;
    var allowed = allowed_in;
    for (var attempt = 0u; attempt < 4u; attempt++) {
        var k = 0u;
        for (var p = 0u; p < P.npts; p++) {
            if (p < P.base || ((allowed >> (p - P.base)) & 1u) == 1u) {
                PIDX[k] = p;
                k++;
            }
        }
        for (var a = 0u; a < k; a++) {
            let pa = pts[PIDX[a]].xyz;
            for (var b = 0u; b < k; b++) {
                G[a][b] = dot(pa, pts[PIDX[b]].xyz) + s2;
            }
            C[a] = dot(pa, t) + s2;
            G[a][a] += P.mu;
            C[a] += P.mu * WBAR[PIDX[a]];
        }
        nnls(k);
        var sum = 0.0;
        for (var a = 0u; a < k; a++) {
            sum += X[a];
        }
        for (var p = 0u; p < MAXP; p++) {
            M[p] = 0.0;
        }
        if (sum > 0.0) {
            for (var a = 0u; a < k; a++) {
                M[PIDX[a]] = X[a] / sum;
            }
        } else {
            M[0] = 1.0;
        }
        // 层数超限:只保留权重最大的 max_layers 种油墨重解
        var used = 0u;
        for (var p = P.base; p < P.npts; p++) {
            if (M[p] > 1e-4) {
                used++;
            }
        }
        if (used <= P.max_layers) {
            return;
        }
        var keep = 0u;
        for (var r = 0u; r < P.max_layers; r++) {
            var bp = 0u;
            var bw = -1.0;
            for (var p = P.base; p < P.npts; p++) {
                let bit = 1u << (p - P.base);
                if ((keep & bit) == 0u && M[p] > 1e-4 && M[p] > bw) {
                    bw = M[p];
                    bp = p;
                }
            }
            keep |= 1u << (bp - P.base);
        }
        allowed = keep;
    }
}

fn coverage_at(p: u32) -> f32 {
    var below = 0.0;
    for (var i = 0u; i <= p; i++) {
        below += M[i];
    }
    if (below > 1e-6) {
        return clamp(M[p] / below, 0.0, 1.0);
    }
    return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var x: u32;
    var ly: u32;
    if (P.mode == 0u) {
        x = gid.x;
        ly = gid.y;
    } else {
        x = gid.x * P.step;
        ly = gid.y * P.step;
    }
    let y = P.y0 + ly;
    if (x >= P.w || ly >= P.rows || y >= P.h) {
        return;
    }
    let li = ly * P.w + x;
    let u = uidx[li];
    for (var p = 0u; p < MAXP; p++) {
        M0[p] = mixes[u * MAXP + p];
    }
    let t = px_at(x, y);
    sample_prior(x, y, smooth_lab(x, y));

    // 颜色最准的解已经和邻域一致:约束不起作用,直接用
    var same = true;
    for (var p = 0u; p < MAXP; p++) {
        if (abs(M0[p] - WBAR[p]) >= 0.02) {
            same = false;
        }
    }
    if (same) {
        for (var p = 0u; p < MAXP; p++) {
            M[p] = M0[p];
        }
    } else {
        var allowed = 0u;
        for (var k = 0u; k < P.n_inks; k++) {
            if (M0[P.base + k] > 1e-4 || WBAR[P.base + k] > 0.02) {
                allowed |= 1u << k;
            }
        }
        solve_reg(t, allowed);
    }

    if (P.mode == 1u) {
        let sw = (P.w + P.step - 1u) / P.step;
        let o = (gid.y * sw + gid.x) * MAXP;
        for (var p = 0u; p < MAXP; p++) {
            outb[o + p] = bitcast<u32>(M[p]);
        }
        atomicAdd(&done[gid.y], 1u);
        return;
    }

    // mode 0:各色版网点值打包输出(槽 0..n_inks-1 = 油墨,槽 n_inks = 白墨打底)
    var words = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
    for (var k = 0u; k < P.n_inks; k++) {
        var b = u32(floor(coverage_at(P.base + k) * 255.0 + 0.5));
        if (b < P.min_dot) {
            b = 0u;
        }
        words[k / 4u] |= b << ((k % 4u) * 8u);
    }
    if (P.base > 1u) {
        let b = u32(floor(coverage_at(1u) * 255.0 + 0.5));
        let s = P.n_inks;
        words[s / 4u] |= b << ((s % 4u) * 8u);
    }
    let o = li * P.out_words;
    for (var i = 0u; i < P.out_words; i++) {
        outb[o + i] = words[i];
    }
    atomicAdd(&done[gid.y], 1u);
}
