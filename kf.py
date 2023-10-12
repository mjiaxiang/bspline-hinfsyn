import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cvxpy

# Define state transition model for Dubin's car
dt = 0.1  # Time step
v = np.array([1.0, 0.5])   # 2-D Constant velocity
theta_dot_max = 0.1  # Maximum angular velocity 

# State transition matrix A
def state_transition(x, u):
    theta = x[2]
    x_new = np.array([
        x[0] + v[0] * np.cos(theta) * dt,
        x[1] + v[1] * np.sin(theta) * dt,
        x[2] + u[1] * dt  # u[1] =  angular velocity
    ])
    return x_new

# Jacobian of the state transition function with respect to state
def jacobian_state(x, u):
    theta = x[2]
    A = np.array([
        [1, 0, -v[0] * np.sin(theta) * dt],
        [0, 1, v[1] * np.cos(theta) * dt],
        [0, 0, 1]
    ])
    return A

# Jacobian of the state transition function with respect to control input
def jacobian_control(x, u):
    theta = x[2]
    B = np.array([
        [np.cos(theta) * dt, 0],
        [np.sin(theta) * dt, 0],
        [0, dt]
    ])
    return B

def error_ell(covariance, center, scale=2):
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate the angle of the major axis (the first eigenvector)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Calculate the standard deviations along the major and minor axes
    std_dev_major = np.sqrt(eigenvalues[0])
    std_dev_minor = np.sqrt(eigenvalues[1])

    # Semi Major and Semi Minor Axis Calculation
    width = scale * std_dev_major
    height = scale * std_dev_minor

    sminor = width / 2
    smajor = height / 2

    # Print Axis Values
    print('Semi Minor Axis:', sminor)
    print('Semi Major Axis:', smajor)
    print('Angle:', angle)

    return sminor, smajor, angle


def plot_ee(sminor, smajor, angle, mean):
    
    # Plot Error Ellipse
    fig, ax = plt.subplots(subplot_kw={'aspect':'equal'})
    ell = Ellipse(xy=mean, width = sminor * 2, height = smajor * 2, angle=angle)
    ax.add_artist(ell)
    ax.set_xlim(mean[0] - 0.20, mean[0] + 0.20)
    ax.set_ylim(mean[1] - 0.20, mean[1] + 0.20)  # Fixed the y-axis limit

    # Plot Semi-Major and Semi-Minor Axis
    ax.axhline(y=mean[1], color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=mean[0], color='gray', linestyle='--', linewidth=1)  # Fixed to axvline for x-axis
    ax.plot([mean[0], mean[0] + (sminor)*np.cos(np.radians(angle))], [mean[1], mean[1] + (sminor)*np.sin(np.radians(angle))], color='red', linestyle='-', linewidth=1)
    ax.plot([mean[0], mean[0] - (smajor)*np.sin(np.radians(angle))], [mean[1], mean[1] + (smajor)*np.cos(np.radians(angle))], color='yellow', linestyle='-', linewidth=1)  # Fixed the sign of y-component

    # Grids
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Error Ellipse From Covariance Matrix')
    plt.show()

def hinfsyn(A, B, C, D, covariance):
    # Define the optimization variables
    n = A.shape[1]  # Number of states
    m = B.shape[1]  # Number of control inputs

    # Define the H-infinity controller gain as a variable
    K = cvxpy.Variable((m, n))

    # Define the disturbance weighting matrix (tunable parameter to fit problem reqs)
    W = cvxpy.Parameter((n, n), PSD=True)  # Has to be positive semi-definite

    # Define the covariance matrix Q (representing uncertainties)
    Q = cvxpy.Parameter((n, n), PSD=True)

    # Define the objective: minimize the worst-case H-infinity norm
    objective = cvxpy.Minimize(np.norm(np.sqrt(Q) @ (np.eye(n) - A @ K), 'hinf'))

    # Define constraints
    constraints = [
        K @ B == C @ K + D,
        K >= 0,
        W >> 0,  # W is positive semidefinite
        cvxpy.bmat([[W, cvxpy.sqrt(Q) @ (A @ K - B)], [cvxpy.transpose(cvxpy.sqrt(Q) @ (A @ K - B)), W]]) >> 0  # New robustness constraint
        # Ensure that the closed-loop system remains stable and robust to uncertainties represented by Q
    ]
    # lhs_1 = np.block([[A.T @ X + X @ A - X @ B @ np.linalg.inv(R) @ B.T @ X + Q, X @ C.T - X @ D],
    #               [C @ X - D @ X @ B.T, -Y]])
    # lhs_2 = np.block([[Y, np.zeros((p, n))],
    #                 [np.zeros((n, p)), X]])
    # lhs = np.block([[lhs_1, np.zeros((lhs_1.shape[0], p))],
    #                 [lhs_2]])
    # rhs = np.zeros(lhs.shape)


    # Formulate the optimization problem
    problem = cvxpy.Problem(objective, constraints)

    # Assign values to the parameters (Q and W)
    Q.value = covariance
    W.value = np.eye(n)

    # Solve the problem
    problem.solve()

    # The optimized H-infinity controller gain is in K.value
    return K.value

# Observation model 
C = np.array([[1, 0, 0],
              [0, 1, 0]])

# 
p = C.shape[0]  # Number of outputs or measurements
m = 2 # Number of control inputs
# m = B.shape[1]
D = np.zeros((p, m))

# Process noise covariance matrix Q
# Change Values
Q = np.diag([0.001, 0.001, 0.001])    

# Measurement noise covariance matrix R
# Change Values
# R = np.diag([0.02826, 0.004312])    # Diagonal Covariance Matrix Instance
R = np.array([[0.02826, 0.004312], [0.004312, 0.004312]])  # Full Covariance Matrix Instance

# Initial state estimate (1D): [x, y, theta]
# Change Values
x_hat = np.array([0, 0, 0])    

# Initial error covariance matrix
# Change Values
P = np.diag([0.001, 0.001, 0.001])

# Simulated measurements (1D)
# Change Values
measurements = np.array([[1.2, 1.9], [3.1, 3.8], [5.0, 6.2]])

# Lists to store estimated states and covariances for plotting
estimated_states = []
estimated_covariances = []

# Kalman filter loop
for i, z in enumerate(measurements):

    # State prediction step
    A = jacobian_state(x_hat, None)
    B = jacobian_control(x_hat, None)
    x_hat_minus = state_transition(x_hat, [v, theta_dot_max])
    P_minus = np.dot(np.dot(A, P), A.T) + Q

    # Compute the control input based on the A matrix and H-inf Controller Gain for current covariance
    K_hinfty = hinfsyn(A, B, C, D, P)
    u_hinfty = -np.dot(K_hinfty, x_hat)

    # Measurement update step
    L = np.dot(np.dot(P_minus, C.T), np.linalg.inv(np.dot(np.dot(C, P_minus), C.T) + R))
    x_hat = x_hat_minus + np.dot(L, z - np.dot(C, x_hat_minus))
    P = np.dot(np.eye(3) - np.dot(L, C), P_minus)  # Update the covariance matrix

    # Append the estimated state and covariance to the lists for plotting
    estimated_states.append(x_hat.copy())
    estimated_covariances.append(P.copy())

    # Print the estimated state (x, y, theta), estimated covariance, and semi major and semi minor axis values
    print('----------------------------------------------------')
    print('Iteration: ', i)
    print("Estimated State (x, y, theta):", x_hat[i])
    print("Estimated Covariance: ", P[i])
    sminor, smajor, estimated_angle = error_ell(estimated_covariances[i], estimated_states[i][:2])
    print('H-Infinity Controller: ', u_hinfty)

# Convert the lists of estimated states and covariances to NumPy arrays for easy plotting
estimated_states = np.array(estimated_states)
estimated_covariances = np.array(estimated_covariances)

# Plot the estimated states
plt.figure(figsize=(10, 6))
plt.plot(estimated_states[:, 0], label='Estimated x')
plt.plot(estimated_states[:, 1], label='Estimated y')
plt.plot(estimated_states[:, 2], label='Estimated theta')
plt.xlabel('Time Step')
plt.ylabel('Estimated State')
plt.legend()
plt.title('Kalman Filter Estimated State')
plt.grid(True)
plt.show()

# Plot the covariance matrix elements
plt.figure(figsize=(10, 6))
plt.plot(estimated_covariances[:, 0, 0], label='Covariance (x, x)')
plt.plot(estimated_covariances[:, 1, 1], label='Covariance (y, y)')
plt.plot(estimated_covariances[:, 2, 2], label='Covariance (theta, theta)')
plt.xlabel('Time Step')
plt.ylabel('Covariance')
plt.legend()
plt.title('Kalman Filter Estimated Covariance')
plt.grid(True)
plt.show()

# Plot the error ellipse based on the estimated covariance matrix
# Example Usage for Error Ellipse
for i in range(len(estimated_covariances)):
    plot_ee(sminor, smajor, estimated_angle, estimated_states[i][:2])