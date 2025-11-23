"""
Default configuration for Multi-Agent football environment scenarios

This module provides a function to generate the configuration dictionary
for any Multi-Agent setup, with override options for key parameters
"""

def get_config(
    fps: int = 24,
    seconds: int = 10,
    n_attackers: int = 2,
    n_defenders: int = 1,
    include_goalkeeper: bool = False,
    randomize_positions: bool = True,
) -> dict:
    """
    Return a fully constructed configuration dictionary for the multi-agent football environment

    Args:
        fps (int): Simulation frame rate (frames per second)
        seconds (int): Episode duration in seconds
        n_attackers (int): Number of attacking agents
        n_defenders (int): Number of defending agents
        include_goalkeeper (bool): Whether to include a goalkeeper in the defending team
    Returns:
        dict: Complete environment configuration.
    """
    return {
        "fps": fps,
        "seconds": seconds,
        "time_step": 1 / fps,
        "max_steps": int(fps * seconds),
        "n_attackers": n_attackers,
        "n_defenders": n_defenders,
        "include_goalkeeper": include_goalkeeper,
        "randomize_positions": randomize_positions
    }
