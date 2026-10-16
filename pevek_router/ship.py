"""Модель судна: сопротивление (тихая вода + ветер + волны + лёд) -> мощность -> скорость и расход топлива.

Все функции векторизованы по numpy: на вход можно подавать целые поля (2D-массивы ячеек).
Единицы: скорость м/с, сила Н, мощность Вт, толщина м, сплочённость 0..1, волна Hs м.

Упрощения (для пилота, подлежат калибровке по ледовому паспорту и данным судна):
  * тихая вода — адмиралтейская зависимость P ~ V^3 от точки сдаточной скорости;
  * ветер — аэродинамическое сопротивление надстройки (Blendermann-подобная форма, cos-распределение);
  * волны — STAWAVE-1 (ITTC 7.5-02-07-02.2) для встречного волнения с угловым множителем;
  * лёд — двухчленная формула для битого льда (ломка/притапливание + импульс),
    по структуре близкая к Colbourne (2000) / Lindqvist (1989); коэффициенты подобраны так,
    чтобы при мощности ~5 МВт судно в разреженном льду 0.3 м держало 8–10 уз до практических ограничений.
  * добровольное снижение скорости на встречной волне Hs > 3.5 м (слеминг, заливаемость).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KN = 0.514444  # м/с в узле
RHO_AIR = 1.225
RHO_WATER = 1025.0
RHO_ICE = 917.0
G = 9.81


@dataclass
class ShipParams:
    name: str = "Сухогруз ледового класса Л2: ледопроходимость до 0,5 м, до 5 баллов"
    ice_class: str = "Л2"
    L: float = 130.0            # длина между перпендикулярами, м
    B: float = 17.3             # ширина, м
    T: float = 7.0              # осадка, м
    A_front: float = 215.0      # лобовая парусность, м^2
    C_air: float = 0.85         # аэродинамический коэффициент при встречном ветре
    mcr_kw: float = 5300.0      # MCR главного двигателя, кВт
    max_load: float = 0.90      # допустимая длительная нагрузка от MCR
    v_service_kn: float = 14.0  # сдаточная скорость на тихой воде
    p_service_frac: float = 0.85  # доля MCR на сдаточной скорости
    eta_d: float = 0.65         # пропульсивный КПД
    sfoc_g_kwh: float = 185.0   # удельный расход при оптимальной нагрузке
    aux_t_day: float = 1.5      # вспомогательные дизели, т/сут
    # Ледовые ограничения для класса Л2 (самостоятельное плавание в мелкобитом разреженном льду)
    ice_c_max: float = 0.50     # сплочённость (5 баллов), выше — только с ледоколом
    ice_h_max: float = 0.50     # толщина, м (ледопроходимость по ТЗ)
    hs_max: float = 6.0         # предельная значительная высота волны, м
    # Коэффициенты ледового сопротивления
    k_break: float = 8.0
    k_moment: float = 12.0
    c_wave: float = 2.5         # калибровочный множитель STAWAVE-1 (метод занижает потери на крутой волне)
    # Ограничения скорости во льду (практика ледового плавания для слабых классов), узлы
    ice_speed_limits: tuple = field(default_factory=lambda: ((0.10, 99.0), (0.30, 9.0), (0.50, 6.0)))

    @property
    def v_service(self) -> float:
        return self.v_service_kn * KN

    @property
    def p_service_w(self) -> float:
        return self.mcr_kw * 1e3 * self.p_service_frac

    @property
    def p_max_w(self) -> float:
        return self.mcr_kw * 1e3 * self.max_load


# ---------------------------------------------------------------- компоненты сопротивления

def r_calm(ship: ShipParams, v):
    """Сопротивление на тихой воде, Н. P = R*V/eta = P_s*(V/V_s)^3."""
    v = np.asarray(v, float)
    r_s = ship.p_service_w * ship.eta_d / ship.v_service
    return r_s * (v / ship.v_service) ** 2


def r_wind(ship: ShipParams, v, wind_speed, wind_rel_angle):
    """Добавочное сопротивление от ветра, Н.

    wind_rel_angle — угол между курсом и направлением, ОТКУДА дует ветер (0 = встречный), рад.
    Считаем по кажущемуся ветру и вычитаем сопротивление воздуха в штиль (оно уже в r_calm).
    """
    v = np.asarray(v, float)
    u = np.asarray(wind_speed, float)
    th = np.asarray(wind_rel_angle, float)
    # кажущийся ветер: продольная и поперечная компоненты
    ua_x = v + u * np.cos(th)
    ua_y = u * np.sin(th)
    ua = np.hypot(ua_x, ua_y)
    beta = np.arctan2(ua_y, ua_x)
    cx = ship.C_air * np.cos(beta)
    r = 0.5 * RHO_AIR * ship.A_front * (cx * ua ** 2 - ship.C_air * v ** 2)
    return r


def r_waves(ship: ShipParams, hs, wave_rel_angle):
    """Добавочное сопротивление на волнении, Н (STAWAVE-1 с угловым множителем).

    wave_rel_angle — угол между курсом и направлением, ОТКУДА идут волны (0 = встречные), рад.
    """
    hs = np.asarray(hs, float)
    th = np.asarray(wave_rel_angle, float)
    r_head = RHO_WATER * G * hs ** 2 * ship.B * np.sqrt(ship.B / ship.L) / 16.0
    f = np.clip(0.5 * (1.0 + np.cos(th)), 0.0, 1.0) ** 1.5  # попутная волна почти не тормозит
    return ship.c_wave * r_head * f


def r_ice(ship: ShipParams, v, conc, thick):
    """Сопротивление битого льда, Н. conc 0..1, thick м."""
    v = np.asarray(v, float)
    c = np.clip(np.asarray(conc, float), 0.0, 1.0)
    h = np.clip(np.asarray(thick, float), 0.0, None)
    c_eff = np.where(c < 0.10, 0.0, c)  # отдельные льдины (<1 балла) не учитываем
    r = c_eff ** 2 * (ship.k_break * RHO_ICE * G * h ** 2 * ship.B
                      + ship.k_moment * RHO_ICE * ship.B * h * v ** 2)
    return r


def total_resistance(ship, v, wind_speed, wind_rel, hs, wave_rel, conc, thick):
    return (r_calm(ship, v) + r_wind(ship, v, wind_speed, wind_rel)
            + r_waves(ship, hs, wave_rel) + r_ice(ship, v, conc, thick))


def shaft_power(ship, v, *env):
    """Требуемая мощность на валу, Вт (не меньше нуля)."""
    r = total_resistance(ship, v, *env)
    return np.maximum(r * v / ship.eta_d, 0.0)


def sfoc(ship: ShipParams, load):
    """Удельный расход топлива, г/кВт·ч, как функция нагрузки (минимум около 80 % MCR)."""
    load = np.clip(np.asarray(load, float), 0.1, 1.0)
    return ship.sfoc_g_kwh * (1.0 + 0.9 * (load - 0.8) ** 2)


def ice_speed_cap(ship: ShipParams, conc):
    """Предельная скорость во льду, м/с (практические ограничения для слабого ледового класса)."""
    c = np.asarray(conc, float)
    cap = np.full(c.shape, 99.0)
    lo = 0.0
    for c_hi, v_kn in ship.ice_speed_limits:
        cap = np.where((c >= lo) & (c < c_hi), v_kn, cap)
        lo = c_hi
    cap = np.where(c >= lo, ship.ice_speed_limits[-1][1], cap)
    return cap * KN


def navigable(ship: ShipParams, conc, thick, hs):
    """Маска допустимого самостоятельного плавания судна данного класса."""
    c = np.asarray(conc, float)
    h = np.asarray(thick, float)
    # ледопроходимость: сплочённость не выше ice_c_max и толщина не больше ice_h_max (отдельные льдины < 1 балла не учитываем)
    ok_ice = (c <= ship.ice_c_max + 1e-6) & ((c < 0.10) | (h <= ship.ice_h_max))
    return ok_ice & (np.asarray(hs, float) <= ship.hs_max)


def max_speed(ship, wind_speed, wind_rel, hs, wave_rel, conc, thick, iters=30):
    """Максимальная скорость по воде при ограничении мощности (бисекция, векторно), м/с."""
    shape = np.broadcast(wind_speed, wind_rel, hs, wave_rel, conc, thick).shape
    lo = np.zeros(shape)
    hi = np.full(shape, ship.v_service * 1.15)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = shaft_power(ship, mid, wind_speed, wind_rel, hs, wave_rel, conc, thick)
        ok = p <= ship.p_max_w
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return np.minimum(np.minimum(lo, ice_speed_cap(ship, conc)), wave_speed_cap(ship, hs, wave_rel))


def wave_speed_cap(ship, hs, wave_rel):
    """Добровольное снижение скорости на встречно-скуловом волнении, м/с."""
    hs = np.asarray(hs, float)
    head = np.clip(np.cos(np.asarray(wave_rel, float)), 0.0, 1.0)
    frac = 1.0 - 0.10 * np.clip(hs - 3.5, 0.0, None) * head
    return np.clip(frac, 0.4, 1.0) * ship.v_service * 1.15


def fuel_rate_t_per_h(ship, power_w):
    """Расход топлива, т/ч: главный двигатель + вспомогательные."""
    p_kw = np.asarray(power_w, float) / 1e3
    load = p_kw / ship.mcr_kw
    main = p_kw * sfoc(ship, load) / 1e6
    return main + ship.aux_t_day / 24.0


L2_SHIP = ShipParams()
