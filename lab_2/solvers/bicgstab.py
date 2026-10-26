"""Unpreconditioned BiCGSTAB with true residual monitoring."""

from sys import float_info

from linalg import Matrix, Vector, dot, matvec, norm
from solvers.base import LinearSolver
from solvers.common import IterationState, Options, Result


def _near_orthogonal(value: float, x: Vector, y: Vector) -> bool:
    return abs(value) <= 16 * float_info.epsilon * norm(x) * norm(y)


class BiCGSTABSolver(LinearSolver):
    method = "bicgstab"

    def _solve(self, a: Matrix, f: Vector, options: Options) -> Result:
        state = IterationState("bicgstab", a, f, options)
        initial = state.initial_result()
        if initial:
            return initial
        r, shadow = f[:], f[:]
        p, v = [0.0] * len(f), [0.0] * len(f)
        rho_old = alpha = omega = 1.0
        fresh = True
        for _ in range(options.max_iterations):
            rho = dot(shadow, r)
            if _near_orthogonal(rho, shadow, r):
                return state.finish("breakdown", "BiCGSTAB: (r_shadow,r) is close to zero.")
            if fresh:
                p = r[:]
                fresh = False
            else:
                if omega == 0.0:
                    return state.finish("breakdown", "BiCGSTAB: omega=0.")
                beta = (rho / rho_old) * (alpha / omega)
                p = [ri + beta * (pi - omega * vi) for ri, pi, vi in zip(r, p, v)]
            v = matvec(a, p)
            denominator = dot(shadow, v)
            if _near_orthogonal(denominator, shadow, v):
                return state.finish("breakdown", "BiCGSTAB: (r_shadow,Ap) is close to zero.")
            alpha = rho / denominator
            s = [ri - alpha * vi for ri, vi in zip(r, v)]
            half_x = [xi + alpha * pi for xi, pi in zip(state.x, p)]
            if norm(s) <= state.threshold:
                actual_r, status = state.advance(half_x)
                if status:
                    return state.finish(status)
                r, shadow, fresh = actual_r, actual_r[:], True
                rho_old = alpha = omega = 1.0
                continue
            t = matvec(a, s)
            tt = dot(t, t)
            if tt == 0.0:
                return state.finish("breakdown", "BiCGSTAB: (As,As)=0 with a nonzero residual.")
            omega = dot(t, s) / tt
            if omega == 0.0:
                return state.finish("breakdown", "BiCGSTAB: the stabilization step omega=0.")
            x = [xi + omega * si for xi, si in zip(half_x, s)]
            actual_r, status = state.advance(x)
            if status:
                return state.finish(status)
            r = [si - omega * ti for si, ti in zip(s, t)]
            if norm(r) <= state.threshold:
                r, shadow, fresh = actual_r, actual_r[:], True
                rho_old = alpha = omega = 1.0
            else:
                rho_old = rho
        return state.finish("iteration_limit")
