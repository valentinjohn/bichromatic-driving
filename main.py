import sys
import subprocess

PYTHON = sys.executable

def main():
    print("Running script for Figure 1...")
    subprocess.run([PYTHON, "figure1.py"], check=True)

    print("Running script for Figure 2...")
    subprocess.run([PYTHON, "figure2.py"], check=True)

    print("Running script for Figure 3...")
    subprocess.run([PYTHON, "figure3.py"], check=True)

    print("Running script for Figure 4...")
    subprocess.run([PYTHON, "figure4.py"], check=True)

    print("Running script for Figure S1...")
    subprocess.run([PYTHON, "figureS1.py"], check=True)

    print("Running script for Figure S2...")
    subprocess.run([PYTHON, "figureS2.py"], check=True)

    print("Running script for Figure S3...")
    subprocess.run([PYTHON, "figureS3.py"], check=True)

    print("Running script for Figure S4...")
    subprocess.run([PYTHON, "figureS4.py"], check=True)

    print("Running script for Figure S5...")
    subprocess.run([PYTHON, "figureS5.py"], check=True)

if __name__ == '__main__':
    main()
