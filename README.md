# SafeExploring

SafeExploring is an application that utilizes reinforcement learning, specifically SARSA, to train a robotic agent in two scenarios: Cliffwalking and Cart Pole. The primary objective is to reduce the number of times the agent falls into dangerous states in the original scenario. This is achieved by incorporating the concept of "contextual affordance," which establishes a relationship between an action, a state, and a location to predict if the action will result in a negative outcome, thereby mitigating potential harm. To facilitate this, a neural network operates as an external guide, providing real-time support to the agent as it navigates these scenarios.

## Workflow
- The robotic agent explores the controlled scenario (Cliffwalking or Cart Pole) using the SARSA algorithm to collect training data.
- The collected data is used to train an artificial neural network with a multilayer perceptron.
- The neural network acts as an external guide for the agent, applying contextual affordance to predict whether an action will lead the agent into a dangerous state.
- Once the network is trained, the agent is transferred to the original scenario and uses the support provided by the neural network to reduce falls into dangerous states.

## Usage
To use SafeExploring, follow these steps:

- Clone the SafeExploring repository.
- Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```
- Run the application in your development environment using:
```bash
python3 main.py
```
## Contribution
If you wish to contribute to SafeExploring, follow these steps:

- Fork the repository.
- Create a branch for your contribution.
- Make your changes and ensure everything works correctly.
- Submit a pull request for review.

## License

[MIT](https://choosealicense.com/licenses/mit/)
