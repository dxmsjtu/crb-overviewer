% MATLAB example for Quadratic Least Squares for peak detection
% MATLAB code for High Precision Clock Synchronization using Sinc-NLS (without fminunc)
% MATLAB code for High Precision Clock Synchronization using Sinc-NLS

 clear all; close all; 
 En_fminunc = 0;
 if En_fminunc = 2;
% Generate noisy data (sine wave + noise)
Fs = 1e3; % Sampling frequency (1 kHz)
T = 1; % Duration of signal (1 second)
t = 0:1/Fs:T-1/Fs; % Time vector
true_peak_position = 0.5; % True peak at t = 0.5 seconds
y = sin(2 * pi * 10 * t) + 0.1 * randn(size(t)); % Sine wave with noise

% Approximate the peak position using Quadratic Least Squares (QLS)
% Select a small window around the peak to fit a quadratic function
window_size = 30; % Number of samples around the peak to use for fitting
peak_index = round(length(t)/2); % Assume peak near the middle

% Define a small window around the peak
t_window = t(peak_index - window_size : peak_index + window_size);
y_window = y(peak_index - window_size : peak_index + window_size);

% Fit a quadratic function: y(t) = a * t^2 + b * t + c
X = [t_window.^2, t_window, ones(size(t_window))]; % Design matrix
params = (X' * X) \ (X' * y_window'); % Solve for coefficients [a; b; c]

% Extract the coefficients
a = params(1);
b = params(2);
c = params(3);

% Calculate the peak position (vertex of the parabola)
t_peak = -b / (2 * a); % Vertex formula for quadratic function

% Plot the results
figure;
plot(t, y, 'k', 'LineWidth', 1.5);
hold on;
plot(t_window, X * params, 'r--', 'LineWidth', 2); % Plot the quadratic fit
xlabel('Time (s)');
ylabel('Amplitude');
legend('Noisy Signal', 'Quadratic Fit');
title(sprintf('Peak Detection: Estimated Peak at t = %.4f seconds', t_peak));
grid on;
 end


% MATLAB code for High Precision Clock Synchronization using Sinc-NLS (without fminunc)
clear all; close all; 

if En_fminunc ==0 %  (without fminunc)
% MATLAB code for High Precision Clock Synchronization using Sinc-NLS (without fminunc)
% Step 1: Generate a simulated signal (ideal signal + noise)
Fs = 1e9; % Sampling frequency (1 GHz)
T = 1e-6;  % Signal duration (1 microsecond)
t = 0:1/Fs:T-1/Fs; % Time vector
true_peak_position = 5e-8; % True peak position (50 ns)
tau = 1e-9; % Time width of the signal (1 ns)
A = 1; % Amplitude of the signal

% Generate a noisy Sinc signal with added noise
signal = A * sinc((t - true_peak_position) / tau);
noise = 0.1 * randn(size(t)); % Additive Gaussian noise
noisy_signal = signal + noise;

% Step 2: Define the Sinc function model for fitting
sinc_model = @(params, t) params(1) * sinc((t - params(2)) / params(3)); % Sinc model

% Step 3: Define the objective function for least squares fitting
% params = [Amplitude, Peak Position (t0), Time width (tau)]
objective_function = @(params) sum((noisy_signal - sinc_model(params, t)).^2);

% Step 4: Define the gradient of the objective function
% Gradient of the least squares error with respect to [A, t0, tau]
gradient_function = @(params) [
    -2 * sum((noisy_signal - sinc_model(params, t)) .* sinc((t - params(2)) / params(3))); % dL/dA
    2 * sum((noisy_signal - sinc_model(params, t)) .* params(1) * (t - params(2)) .* sinc((t - params(2)) / params(3)).^2 / params(3)); % dL/dt0
    2 * sum((noisy_signal - sinc_model(params, t)) .* params(1) * (t - params(2)).^2 .* sinc((t - params(2)) / params(3)).^2 / params(3).^2); % dL/dtau
];

% Step 5: Implement the Gradient Descent algorithm
% Initialize parameters
initial_guess = [A, 5e-8, 1e-9]; % Initial guess for [Amplitude, Peak position (t0), Time width (tau)]
learning_rate = 1e5; % Learning rate for gradient descent
max_iterations = 200; % Maximum number of iterations
tolerance = 1e-8; % Convergence tolerance

params = initial_guess; % Start with the initial guess
previous_loss = objective_function(params); % Calculate the initial loss

% Perform gradient descent
for iter = 1:max_iterations
    grad = gradient_function(params); % Compute the gradient
    params = params - learning_rate * grad; % Update the parameters using the gradient
    
    % Calculate the new loss
    current_loss = objective_function(params);
    
    % Check for convergence (if the loss change is very small)
    if abs(previous_loss - current_loss) < tolerance
        fprintf('Converged after %d iterations.\n', iter);
        break;
    end
    
    previous_loss = current_loss; % Update the loss for the next iteration
end

% Step 6: Extract the estimated parameters
estimated_amplitude = params(1);
estimated_peak_position = params(2);
estimated_tau = params(3);

% Step 7: Display results
fprintf('True Peak Position: %.8f s\n', true_peak_position);
fprintf('Estimated Peak Position: %.8f s\n', estimated_peak_position);
fprintf('Estimated Amplitude: %.4f\n', estimated_amplitude);
fprintf('Estimated Time Width (tau): %.8f s\n', estimated_tau);

% Step 8: Plot the results
figure;
plot(t * 1e9, noisy_signal, 'k', 'LineWidth', 1.5); % Plot noisy signal
hold on;
plot(t * 1e9, sinc_model(params, t), 'r--', 'LineWidth', 2); % Plot fitted Sinc model
xlabel('Time (ns)');
ylabel('Amplitude');
legend('Noisy Signal', 'Fitted Sinc Model');
title('High Precision Clock Synchronization using Sinc-NLS (Gradient Descent)');
grid on;
end 


% MATLAB code for High Precision Clock Synchronization using Sinc-NLS
clear all; close all; 
% Step 1: Generate a simulated signal (ideal signal + noise)
Fs = 1e9; % Sampling frequency (1 GHz)
T = 1e-6; % Signal duration (1 microsecond)
t = 0:1/Fs:T-1/Fs; % Time vector
true_peak_position = 5e-8; % True peak position (50 ns)
tau = 1e-9; % Time width of the signal (1 ns)
A = 1; % Amplitude of the signal

% Generate a noisy Sinc signal with added noise
signal = A * sinc((t - true_peak_position) / tau);
noise = 0.1 * randn(size(t)); % Additive Gaussian noise
noisy_signal = signal + noise;

% Step 2: Define the Sinc function model for fitting
sinc_model = @(params, t) params(1) * sinc((t - params(2)) / params(3)); % Sinc model

% Step 3: Define the objective function for least squares fitting
% params = [Amplitude, Peak Position (t0), Time width (tau)]
objective_function = @(params) sum((noisy_signal - sinc_model(params, t)).^2);

% Step 4: Initial guess for parameters
initial_guess = [A, 4.5e-8, 1.3e-9]; % Initial guess for Amplitude, Peak position (t0), Time width (tau)

% Step 5: Use fminunc (Nonlinear optimization) to minimize the least squares error
options = optimset('Display', 'off', 'TolFun', 1e-8, 'TolX', 1e-8); % Set optimization options
estimated_params = fminunc(objective_function, initial_guess, options);

% Step 6: Extract the estimated parameters
estimated_amplitude = estimated_params(1);
estimated_peak_position = estimated_params(2);
estimated_tau = estimated_params(3);

% Step 7: Display results
fprintf('True Peak Position: %.8f s\n', true_peak_position);
fprintf('Estimated Peak Position: %.8f s\n', estimated_peak_position);
fprintf('Estimated Amplitude: %.4f\n', estimated_amplitude);
fprintf('Estimated Time Width (tau): %.8f s\n', estimated_tau);

% Step 8: Plot the results
figure;
plot(t * 1e9, noisy_signal, 'k', 'LineWidth', 1.5); % Plot noisy signal
hold on;
plot(t * 1e9, sinc_model(estimated_params, t), 'r--', 'LineWidth', 2); % Plot fitted Sinc model
xlabel('Time (ns)');
ylabel('Amplitude');
legend('Noisy Signal', 'Fitted Sinc Model');
title('High Precision Clock Synchronization using Sinc-NLS');
grid on;