import sys
from football_tactical_ai.helpers.helperTraining import train_SingleAgent

def main():
    
    if len(sys.argv) != 2:
        print("Example: python trainingSingleAgentCatShot shot_weak")
        print("Available: shot_weak, shot_normal, shot_strong")
        sys.exit(1)

    scenario = sys.argv[1]
    train_SingleAgent(scenario)

if __name__ == "__main__":
    main()