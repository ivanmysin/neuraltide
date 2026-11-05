"""
Контрольные варианты к test_audit_regressions.py (написаны при запуске
регрессионных тестов аудита, 3 ноября 2026; диагностика В6 расширена
после исправления В6, ноябрь 2026).

- К2: тот же сценарий с include_syn_states=False и True. С True склейка
  совпадает со свободным прогоном; с False состояния синапсов сбрасываются
  на узлах, и finalize обязан вернуть consistent=False.
- В6: диагностика порядка сходимости с контрольными окнами: Эйлер ~1 во
  всех режимах; RK4 со связью: split ~2 (расщеплённый шаг), network-wide
  > 3.5 (согласованный шаг); RK4 без связи (gsyn = 0): ~4 в обоих режимах.
"""

import numpy as np
import pytest
import tensorflow as tf

import neuraltide.config
from neuraltide.integrators import EulerIntegrator, RK4Integrator
from neuraltide.optimization import (
    CalibrationProblem,
    ObservationSpec,
    ParameterSpace,
    ResidualLoss,
)
from neuraltide.optimization.shooting import MultipleShootingProblem, ShootingConfig

from tests.unit.test_audit_regressions import _ee_tm_network, _t_seq


# ---------------------------------------------------------------------------
# К2: контроль. С include_syn_states=True склейка должна совпасть со
# свободным прогоном — значит, расхождение в основном тесте даёт именно
# сброс состояний синапсов.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:MultipleShootingProblem")
@pytest.mark.parametrize("include_syn", [False, True])
def test_ms_tm_synapse_control(include_syn):
    neuraltide.config.set_dtype(tf.float64)
    dt, n = 0.5, 200
    network = _ee_tm_network(dt, EulerIntegrator())
    t_seq = _t_seq(n, dt)
    target = network(t_seq).firing_rates["exc"]
    loss = ResidualLoss(
        [ObservationSpec(name="obs", population="exc", target=target, sigma=2.0)]
    )
    space = ParameterSpace.from_network(network)
    problem = CalibrationProblem(network, space, t_seq, objective=loss)
    ms = MultipleShootingProblem(
        problem, ShootingConfig(4), include_syn_states=include_syn
    )
    phi = np.asarray(ms.initialize_from_single_shooting(), dtype=np.float64)
    check = ms.finalize(phi, gap_tol=1e-8)
    print(
        f"\ninclude_syn_states={include_syn}: loss_ms={check.loss_ms:.3e} "
        f"loss_single={check.loss_single:.3e} gap_max={check.gaps['max']:.3e} "
        f"consistent={check.consistent}"
    )
    if include_syn:
        assert check.consistent
        assert check.loss_ms == pytest.approx(check.loss_single, rel=1e-6, abs=1e-12)
    else:
        # Сброс синапсов на узлах: склейка расходится со свободным прогоном,
        # и finalize обязан это показать.
        assert not check.consistent
        assert check.loss_ms != pytest.approx(check.loss_single, rel=1e-3)


# ---------------------------------------------------------------------------
# В6: измеренный порядок сети с контрольными окнами. Эйлер ~1 везде;
# RK4 со связью: split ~2, network-wide > 3.5; RK4 без связи: ~4 в обоих.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integ", ["euler", "rk4"])
@pytest.mark.parametrize("mode", ["split", "network"])
@pytest.mark.parametrize("coupled", [True, False])
def test_network_order_diagnostic(integ, mode, coupled):
    neuraltide.config.set_dtype(tf.float64)
    t_end = 20.0
    make = {"euler": EulerIntegrator, "rk4": RK4Integrator}[integ]

    def final_E(dt):
        net = _ee_tm_network(dt, make(), gsyn=0.5 if coupled else 0.0, mode=mode)
        out = net(_t_seq(int(round(t_end / dt)), dt))
        return float(np.asarray(out.final_state[0][0]).ravel()[0])

    ref = final_E(0.5 / 64)
    dts = (0.5, 0.25, 0.125)
    errs = [abs(final_E(dt) - ref) for dt in dts]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print(
        f"\n{integ} mode={mode} coupled={coupled}: "
        f"errs={['%.2e' % e for e in errs]} orders={['%.2f' % o for o in orders]}"
    )
    order = orders[-1]
    if integ == "euler":
        lo, hi = 0.8, 1.3
    elif coupled and mode == "split":
        lo, hi = 1.5, 2.6  # расщеплённый шаг: частота в начале, ток в конце
    else:
        lo, hi = 3.5, 4.6  # согласованный шаг (network) или связь выключена
    assert lo < order < hi, (integ, mode, coupled, order, errs)
