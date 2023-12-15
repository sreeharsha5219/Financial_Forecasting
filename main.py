import numpy as numpy_lib
import pandas as pd
import matplotlib.pyplot as proj_plot
import seaborn as sns

# Load and preprocess the financial data
file_path = 'https://raw.githubusercontent.com/somanathvamshi/Minist_rnn/main/HistoricalData_1698688780156.csv'
market_data = pd.read_csv(file_path)
market_data['Adjusted_Close'] = market_data['Close/Last'].replace('[\$,]', '', regex=True).astype(float)

# Normalize the closing prices
closing_prices = market_data['Adjusted_Close'].values
highest_price = numpy_lib.max(closing_prices)
lowest_price = numpy_lib.min(closing_prices)
scaled_prices = (closing_prices - lowest_price) / (highest_price - lowest_price)

def generate_seq(data_array, length_of_sequence):
    data_sequences, target_values = [], []
    for idx in range(len(data_array) - length_of_sequence):
        sequence_chunk = data_array[idx:(idx + length_of_sequence)]
        target = data_array[idx + length_of_sequence]
        data_sequences.append(sequence_chunk)
        target_values.append(target)
    return numpy_lib.array(data_sequences), numpy_lib.array(target_values)

# Generate input sequences
seq_len = 4  
input_data, output_data = generate_seq(scaled_prices, seq_len)

# Splitting dataset
training_split = int(0.8 * len(input_data))
train_input, test_input = input_data[:training_split], input_data[training_split:]
train_output, test_output = output_data[:training_split], output_data[training_split:]

# Evaluation metrics
def calculate_root_mean_s_error(predicted, actual):
    return numpy_lib.sqrt(numpy_lib.mean((predicted - actual) ** 2))

def calculate_mae(predicted, actual):
    return numpy_lib.mean(numpy_lib.abs(predicted - actual))

def calculate_mape(predicted, actual):
    actual, predicted = numpy_lib.array(actual), numpy_lib.array(predicted)
    not_zero = actual != 0
    return numpy_lib.mean(numpy_lib.abs((actual[not_zero] - predicted[not_zero]) / actual[not_zero])) * 100

# RNN model class
class FinancialForecastModel:
    def __init__(self, s_inp, s_hid, s_out, rate_l=0.01):
        self.s_inp = s_inp
        self.s_hid = s_hid
        self.s_out = s_out
        self.rate_l = rate_l
        self.w_to_hid = numpy_lib.random.randn(s_inp, s_hid) * 0.01
        self.w_h_to_hid = numpy_lib.random.randn(s_hid, s_hid) * 0.01
        self.w_h_to_out = numpy_lib.random.randn(s_hid, s_out) * 0.01
        self.b_h_lay = numpy_lib.zeros((1, s_hid))
        self.b_o_lay = numpy_lib.zeros((1, s_out))

    def logistic_function(self, value):
        return 1 / (1 + numpy_lib.exp(-value))

    def execute_forward_pass(self, inp_seq):
        states_hid = []
        hidden_state = numpy_lib.zeros((self.s_hid,))
        for time_step in range(inp_seq.shape[1]):
            hidden_state = self.logistic_function(numpy_lib.dot(inp_seq[:, time_step], self.w_to_hid) + numpy_lib.dot(hidden_state, self.w_h_to_hid) + self.b_h_lay)
            states_hid.append(hidden_state)
        fin_out = numpy_lib.dot(hidden_state, self.w_h_to_out) + self.b_o_lay
        return fin_out, states_hid

    def optimize_parameters(self, input_set, target_set):
        for input_vector, target_vector in zip(input_set, target_set):
            predicted_output, states_hid = self.execute_forward_pass(input_vector.reshape(1, -1))
            error_value = numpy_lib.mean((predicted_output - target_vector) ** 2)
            grad_w_i_hid, grad_w_h_hid, grad_w_h_out = numpy_lib.zeros_like(self.w_to_hid), numpy_lib.zeros_like(self.w_h_to_hid), numpy_lib.zeros_like(self.w_h_to_out)
            grad_b_hid, grad_b_out = numpy_lib.zeros_like(self.b_h_lay), numpy_lib.zeros_like(self.b_o_lay)
            grad_hid_nxt = numpy_lib.zeros((self.s_hid,))
            for t in reversed(range(len(states_hid))):
                grad_hid = (predicted_output - target_vector) * self.w_h_to_out.T + grad_hid_nxt
                grad_hid_raw = grad_hid * states_hid[t] * (1 - states_hid[t])
                grad_w_i_hid += numpy_lib.outer(input_vector[t], grad_hid_raw)
                grad_w_h_hid += numpy_lib.outer(states_hid[t-1], grad_hid_raw) if t != 0 else numpy_lib.zeros_like(grad_w_h_hid)
                grad_b_hid += grad_hid_raw
                grad_hid_nxt = numpy_lib.dot(grad_hid_raw, self.w_h_to_hid.T)
            self.w_h_to_out -= self.rate_l * grad_w_h_out
            self.w_h_to_hid -= self.rate_l * grad_w_h_hid
            self.w_to_hid -= self.rate_l * grad_w_i_hid
            self.b_o_lay -= self.rate_l * grad_b_out
            self.b_h_lay -= self.rate_l * grad_b_hid

# Hyperparameter settings
hidden_unit_options = [10, 20, 30]
learning_rates = [0.01, 0.001, 0.0001]
epoch_settings = [10, 50, 100]

# Logging experiments
log_experiments = pd.DataFrame(columns=['Experiment Number', 'Hidden Units', 'Learning Rate', 'Epochs', 'Train Data RMSE', 'Test Data RMSE', 'Train Data MAE', 'Test Data MAE', 'Train Data MAPE', 'Test Data MAPE'])

# Tracking experiments
experiment_id = 1
for units_hidden in hidden_unit_options:
    for rate_l in learning_rates:
        for epochs in epoch_settings:
            predictor = FinancialForecastModel(s_inp=1, s_hid=units_hidden, s_out=1, rate_l=rate_l)
            for epoch in range(epochs):
                predictor.optimize_parameters(train_input, train_output)
            
            predictions_train = numpy_lib.array([predictor.execute_forward_pass(x.reshape(1, -1))[0] for x in train_input])
            predictions_test = numpy_lib.array([predictor.execute_forward_pass(x.reshape(1, -1))[0] for x in test_input])

            rmse_data_train = calculate_root_mean_s_error(predictions_train, train_output)
            rmse_data_test = calculate_root_mean_s_error(predictions_test, test_output)
            mae_data_train = calculate_mae(predictions_train, train_output)
            mae_data_test = calculate_mae(predictions_test, test_output)
            mape_data_train = calculate_mape(predictions_train, train_output)
            mape_data_test = calculate_mape(predictions_test, test_output)

            log_experiments.loc[experiment_id] = [experiment_id, units_hidden, rate_l, epochs, rmse_data_train, rmse_data_test, mae_data_train, mae_data_test, mape_data_train, mape_data_test]
            
            experiment_id += 1

log_experiments.to_csv('log_experiments.csv', index=False)

# Visualization of RMSE
proj_plot.figure(figsize=(10, 6))
sns.lineplot(data=log_experiments, x='Epochs', y='Train Data RMSE', label='Train Data RMSE')
sns.lineplot(data=log_experiments, x='Epochs', y='Test Data RMSE', label='Test Data RMSE')
proj_plot.title('RMSE Analysis Across Different Epochs')
proj_plot.xlabel('Epochs')
proj_plot.ylabel('RMSE Value')
proj_plot.legend()
proj_plot.show()

# Visualization of MAE
proj_plot.figure(figsize=(10, 6))
sns.lineplot(data=log_experiments, x='Epochs', y='Train Data MAE', label='Train Data MAE')
sns.lineplot(data=log_experiments, x='Epochs', y='Test Data MAE', label='Test Data MAE')
proj_plot.title('MAE Analysis Over Epochs')
proj_plot.xlabel('Epochs')
proj_plot.ylabel('MAE Value')
proj_plot.legend()
proj_plot.show()

# Visualization of MAPE
proj_plot.figure(figsize=(10, 6))
sns.lineplot(data=log_experiments, x='Epochs', y='Train Data MAPE', label='Train Data MAPE')
sns.lineplot(data=log_experiments, x='Epochs', y='Test Data MAPE', label='Test Data MAPE')
proj_plot.title('MAPE Analysis Across Epochs')
proj_plot.xlabel('Epochs')
proj_plot.ylabel('MAPE (%)')
proj_plot.legend()
proj_plot.show()


proj_plot.figure(figsize=(10, 6))
sns.scatterplot(data=log_experiments, x='Learning Rate', y='Test Data RMSE', hue='Hidden Units')
proj_plot.title('Impact of Learning Rate on Test RMSE')
proj_plot.xscale('log')
proj_plot.xlabel('Learning Rate (log scale)')
proj_plot.ylabel('Test Data RMSE')
proj_plot.legend(title='Hidden Units')
proj_plot.show()

# Hidden Units vs. Test RMSE
proj_plot.figure(figsize=(10, 6))
sns.boxplot(data=log_experiments, x='Hidden Units', y='Test Data RMSE')
proj_plot.title('Effect of Hidden Units on Test RMSE')
proj_plot.xlabel('Number of Hidden Units')
proj_plot.ylabel('Test RMSE Value')
proj_plot.show()
