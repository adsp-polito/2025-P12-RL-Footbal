import os
from football_tactical_ai.helpers.helperTraining import train_SingleAgent

def main():

    #print("\n===== TRAIN: MOVE FAST =====")
    #train_SingleAgent("move_fast")

    print("\n===== TRAIN: MOVE SLOW =====")
    train_SingleAgent("move_slow")

if __name__ == "__main__":
    main()