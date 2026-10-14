"""Heat pump COP calculations based on Hotmaps dispatch methodology."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_heat_pump_cop(
    outdoor_temp: np.ndarray,
    supply_temp: float = 60.0,
    carnot_efficiency: float = 0.50,
    cop_min: float = 2.0,
    cop_max: float = 6.0
) -> np.ndarray:
    """
    Calculate heat pump COP using Hotmaps/Carnot methodology.
    
    Based on: COP = η_carnot * T_sink / (T_sink - T_source)
    
    Args:
        outdoor_temp: Outdoor/source temperature (°C)
        supply_temp: District heating supply temperature (°C)
        carnot_efficiency: Carnot efficiency factor (typical 0.45-0.55)
        cop_min: Minimum COP (technical limit)
        cop_max: Maximum COP (cap for very warm conditions)
    
    Returns:
        Array of COP values for each outdoor temperature
        
    Reference:
        - Hotmaps Heat Roadmap Europe
        - Paardekooper et al. (2018) "Heat Roadmap Europe 4"
    """
    # Convert to Kelvin
    T_source = outdoor_temp + 273.15
    T_sink = supply_temp + 273.15
    
    # Carnot COP
    cop_carnot = T_sink / (T_sink - T_source)
    
    # Actual COP (reduced by Carnot efficiency)
    cop = carnot_efficiency * cop_carnot
    
    # Apply limits
    cop = np.clip(cop, cop_min, cop_max)
    
    return cop


def calculate_ashp_cop_advanced(
    outdoor_temp: np.ndarray,
    supply_temp: float = 60.0,
    heat_pump_type: str = 'air_source'
) -> np.ndarray:
    """
    Advanced heat pump COP calculation with type-specific parameters.
    
    Args:
        outdoor_temp: Outdoor temperature (°C)
        supply_temp: Supply temperature (°C)
        heat_pump_type: 'air_source', 'ground_source', or 'water_source'
    
    Returns:
        COP values
    """
    # Type-specific parameters
    params = {
        'air_source': {
            'carnot_eff': 0.50,
            'cop_min': 2.0,
            'cop_max': 5.5
        },
        'ground_source': {
            'carnot_eff': 0.55,
            'cop_min': 3.0,
            'cop_max': 6.5
        },
        'water_source': {
            'carnot_eff': 0.52,
            'cop_min': 2.5,
            'cop_max': 6.0
        }
    }
    
    p = params.get(heat_pump_type, params['air_source'])
    
    return calculate_heat_pump_cop(
        outdoor_temp,
        supply_temp,
        p['carnot_eff'],
        p['cop_min'],
        p['cop_max']
    )


if __name__ == "__main__":
    print("=" * 80)
    print("HEAT PUMP COP CALCULATIONS - HOTMAPS METHODOLOGY")
    print("=" * 80)
    
    # Test with temperature range
    outdoor_temps = np.arange(-15, 25, 0.5)
    
    # Different supply temperatures
    supply_temps = [50, 60, 70, 80]  # °C
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Heat Pump COP - Hotmaps Methodology', fontsize=16, fontweight='bold')
    
    # 1. COP vs outdoor temperature for different supply temps
    ax = axes[0, 0]
    for t_supply in supply_temps:
        cop = calculate_heat_pump_cop(outdoor_temps, t_supply)
        ax.plot(outdoor_temps, cop, linewidth=2, label=f'Supply: {t_supply}°C')
    
    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('COP (Heat Output / Electric Input)')
    ax.set_title('COP vs. Outdoor Temperature', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='COP=3')
    
    # 2. Different heat pump types
    ax = axes[0, 1]
    for hp_type in ['air_source', 'ground_source', 'water_source']:
        cop = calculate_ashp_cop_advanced(outdoor_temps, 60, hp_type)
        ax.plot(outdoor_temps, cop, linewidth=2, label=hp_type.replace('_', ' ').title())
    
    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('COP')
    ax.set_title('COP by Heat Pump Type (60°C Supply)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Electricity consumption vs temperature
    ax = axes[1, 0]
    heat_output = 100  # kW
    for t_supply in [50, 60, 70]:
        cop = calculate_heat_pump_cop(outdoor_temps, t_supply)
        elec_input = heat_output / cop
        ax.plot(outdoor_temps, elec_input, linewidth=2, label=f'Supply: {t_supply}°C')
    
    ax.set_xlabel('Outdoor Temperature (°C)')
    ax.set_ylabel('Electricity Input (kW) for 100 kW Heat')
    ax.set_title('Electricity Consumption vs. Temperature', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Annual COP distribution
    ax = axes[1, 1]
    
    # Simulate annual temperature profile
    hours = np.arange(8760)
    day_of_year = hours // 24
    temp_annual = 5 + 15 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    
    cop_annual = calculate_heat_pump_cop(temp_annual, 60)
    
    ax.hist(cop_annual, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(cop_annual.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean COP: {cop_annual.mean():.2f}')
    ax.set_xlabel('COP')
    ax.set_ylabel('Hours per Year')
    ax.set_title('Annual COP Distribution (60°C Supply)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('heat_pump_cop_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: heat_pump_cop_analysis.png")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("COP STATISTICS")
    print("=" * 80)
    
    for t_supply in [50, 60, 70]:
        cop = calculate_heat_pump_cop(outdoor_temps, t_supply)
        print(f"\nSupply Temperature: {t_supply}°C")
        print(f"  COP range: {cop.min():.2f} - {cop.max():.2f}")
        print(f"  COP at -10°C: {calculate_heat_pump_cop(np.array([-10]), t_supply)[0]:.2f}")
        print(f"  COP at 0°C: {calculate_heat_pump_cop(np.array([0]), t_supply)[0]:.2f}")
        print(f"  COP at +10°C: {calculate_heat_pump_cop(np.array([10]), t_supply)[0]:.2f}")
    
    print("\n" + "=" * 80)
    print("Annual average COP (60°C supply): {:.2f}".format(cop_annual.mean()))
    print("Annual minimum COP: {:.2f}".format(cop_annual.min()))
    print("Annual maximum COP: {:.2f}".format(cop_annual.max()))
    print("=" * 80)
