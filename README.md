# diabetes-RL-project

This project aims to compare different controllers for (autonomous) insulin infusion. For this purpose the basic implementations of the [T1D simulator](https://github.com/jxx123/simglucose) and [Reinforcement learning algorithms](https://github.com/MHamza-Y/Autonomous-Insulin-Infusion-Controller) were used and adapted. 

## Installation

To install the repository and all required packages (using pip): 
```bash
git clone https://github.com/miopp2/diabetes-RL-project.git
cd simglucose
pip install -r requirements.txt
```

## Results

| BB Controller | PID Controller | PPO with LSTM Policy |
| --- | --- | --- |
| ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/BB_7_days_all.png) | ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/PID_7_days_all.png) | ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/PPO_7_days_all.png)|
| ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/BB_bar_all.png) | ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/PID_bar_all.png) | ![image](https://github.com/miopp2/diabetes-RL-project/imgs_gh/PID_bar_all.png) |

These results were generated by training PPO controller using LstmMlpLnPolicy, T1DDiscreteSimEnv, learning rate of 3e-5, and γ of 0.999. To reproduce these figures run test/simulate_multiple_controllers.py for 7 days (sim_days=7).

## Comments
- if running the code on windows change ''start_method': 'fork'' to ''start_method': 'spawn''
