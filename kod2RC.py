import os
import time
from dotenv import load_dotenv
from iqm.qiskit_iqm import IQMProvider
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# config

METHOD = 'lda' 
N_COMPONENTS = 3  
NUM_QUBITS = N_COMPONENTS

def get_spark_backend(env_file="token.env"):
    load_dotenv(env_file)
    server_url = os.getenv("SERVER", "https://odra5.e-science.pl")
    token = os.getenv("TOKEN")
    provider = IQMProvider(server_url, token=token)
    return provider.get_backend()

dev_sim = qml.device("default.qubit", wires=NUM_QUBITS)

print("Łączenie z komputerem kwantowym IQM...")
iqm_backend = get_spark_backend()
dev_qpu = qml.device(
    "qiskit.remote", wires=NUM_QUBITS, backend=iqm_backend, shots=1000
)


class LDA:
    def __init__(self, n_components=3, reg_param=1e-3):
        self.n_components = n_components
        self.reg_param = reg_param

    def fit(self, X, y):
        classes = np.unique(y)
        n_features = X.shape[1]
        self.mean_overall_ = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            n_c = X_c.shape[0]
            mean_diff = (mean_c - self.mean_overall_).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        S_W += self.reg_param * np.eye(n_features)

        eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(S_W) @ S_B)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        max_lda_components = max(1, len(classes) - 1)
        k_lda = min(self.n_components, max_lda_components)

        self.W_lda_ = np.real(eigvecs[:, :k_lda])

        remaining_dims = self.n_components - k_lda
        if remaining_dims > 0:
            X_c = X - self.mean_overall_
            X_lda_proj = X_c @ self.W_lda_ @ np.linalg.pinv(self.W_lda_)
            X_res = X_c - X_lda_proj

            self.pca_residual_ = PCA(n_components=remaining_dims)
            self.pca_residual_.fit(X_res)
        else:
            self.pca_residual_ = None

        return self

    def transform(self, X):
        X_c = X - self.mean_overall_
        res = [X_c @ self.W_lda_]

        if self.pca_residual_ is not None:
            X_lda_proj = X_c @ self.W_lda_ @ np.linalg.pinv(self.W_lda_)
            X_res = X_c - X_lda_proj
            res.append(self.pca_residual_.transform(X_res))

        return np.hstack(res)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# print("Ładowanie danych")
X_raw, y_raw = load_digits(return_X_y=True)
mask = (y_raw == 0) | (y_raw == 1)

X_filtered, y_filtered = X_raw[mask], y_raw[mask]
y_all = np.where(y_filtered == 1, 1, -1)

X_sub, y_sub = X_filtered, y_all

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_sub, 
    y_sub, 
    test_size=0.2, 
    random_state=42,
    stratify=y_sub
)

print(
    f"Redukcja wymiarowości do {N_COMPONENTS} cech przy użyciu:"
    f" {METHOD.upper()}"
)

if METHOD.lower() == "pca":
    reducer = PCA(n_components=N_COMPONENTS)
    X_train_reduced = reducer.fit_transform(X_train_raw)
    X_test_reduced = reducer.transform(X_test_raw)
elif METHOD.lower() == "lda":
    reducer = LDA(n_components=N_COMPONENTS)
    X_train_reduced = reducer.fit_transform(X_train_raw, y_train)
    X_test_reduced = reducer.transform(X_test_raw)

X_min = X_train_reduced.min(axis=0)
X_max = X_train_reduced.max(axis=0)

range_diff = X_max - X_min
range_diff[range_diff == 0] = 1e-8

X_train = ((X_train_reduced - X_min) / range_diff) * np.pi
X_test = ((X_test_reduced - X_min) / range_diff) * np.pi

# obwód

def circuit_architecture(weights, x):
    qml.AngleEmbedding(
        x,
        wires=range(NUM_QUBITS), 
        rotation='Y'
    )
    
    qml.StronglyEntanglingLayers(
        weights, 
        wires=range(NUM_QUBITS)
    )
    
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_sim)
def quantum_circuit_sim(weights, x):
    return circuit_architecture(weights, x)

@qml.qnode(dev_qpu)
def quantum_circuit_qpu(weights, x):
    return circuit_architecture(weights, x)

def cost_function(weights, X_batch, y_batch):
    predictions = [quantum_circuit_sim(weights, x) for x in X_batch]
    return pnp.mean((pnp.array(predictions) - y_batch) ** 2)

#trening 

num_layers = 2
weights = pnp.array(np.random.randn(num_layers, NUM_QUBITS, 3), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.05)
batch_size = 40
epochs = 8

print("Training ")
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_shuffled = X_train[permutation]
    y_shuffled = y_train[permutation]
    
    for i in range(0, len(X_train), batch_size):
        X_b = X_shuffled[i:i+batch_size]
        y_b = y_shuffled[i:i+batch_size]
        weights, loss = opt.step_and_cost(lambda w: cost_function(w, X_b, y_b), weights)
        
    test_preds_sim = [np.sign(quantum_circuit_sim(weights, x)) for x in X_test]
    accuracy = np.mean(test_preds_sim == y_test)
    print(f"Epoka {epoch+1}/{epochs} | Loss: {loss:.4f} | Sim Test Accuracy: {accuracy * 100:.2f}%")


# Testy -------------------------------------------------------------------


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"wyniki_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

print(f"\nUruchamianie ewaluacji pełnego zbioru testowego na QPU...")
print(f"Wyniki zostaną zapisane w folderze: {results_dir}")

def predict_qpu(x_sc):
    try:
        val = quantum_circuit_qpu(weights, x_sc)
    except Exception as e:
        print(f"Błąd połączenia z QPU: {e}")
        val = 0.0
    time.sleep(0.5)
    return val

test_preds = np.array(
    [1 if predict_qpu(x) >= 0 else -1 for x in X_test]
)

accuracy_qpu = np.mean(test_preds == y_test)
acc_text = f"Końcowa dokładność na QPU (Test Accuracy): {accuracy_qpu * 100:.2f}%"
print(acc_text)

with open(os.path.join(results_dir, "raport_qpu.txt"), "w", encoding="utf-8") as f:
    f.write(f"{acc_text}\n")
    f.write(f"Użyta metoda redukcji: {METHOD.upper()}\n")
    f.write(f"Liczba próbek testowych: {len(y_test)}\n")

# Wykres 1 -----------------------------------------------------

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection="3d")

fig1.suptitle(
    f"Complete Test Set in {METHOD.upper()} Feature Space (3D - QPU)",
    fontsize=14,
    fontweight="bold",
)

mask_pred_0 = test_preds == -1
mask_pred_1 = test_preds == 1

ax1.scatter(
    X_test[mask_pred_0, 0],
    X_test[mask_pred_0, 1],
    X_test[mask_pred_0, 2],
    c="blue",
    marker="o",
    s=35,
    alpha=0.6,
    label="Predicted digit: 0",
)

ax1.scatter(
    X_test[mask_pred_1, 0],
    X_test[mask_pred_1, 1],
    X_test[mask_pred_1, 2],
    c="orange",
    marker="o",
    s=35,
    alpha=0.6,
    label="Predicted digit: 1",
)

incorrect_mask = test_preds != y_test
if np.any(incorrect_mask):
    ax1.scatter(
        X_test[incorrect_mask, 0],
        X_test[incorrect_mask, 1],
        X_test[incorrect_mask, 2],
        c="red",
        marker="x",
        s=70,
        linewidths=2,
        label="Misclassified",
    )

ax1.set_xlabel(f"{METHOD.upper()} Feature 1")
ax1.set_ylabel(f"{METHOD.upper()} Feature 2")
ax1.set_zlabel(f"{METHOD.upper()} Feature 3")

ax1.legend(loc="upper right", fontsize=10)
plt.tight_layout()

fig1.savefig(os.path.join(results_dir, "wykres_3d_qpu.png"), dpi=300, bbox_inches="tight")
plt.show()

# Wykres 2 --------------------------------------------------------------
cm = confusion_matrix(y_test, test_preds, labels=[-1, 1])

fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(cm, cmap='Blues')

ax2.set_title("Confusion Matrix (QPU)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Predicted class")
ax2.set_ylabel("True class")

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])

ax2.set_xticklabels(["0", "1"])
ax2.set_yticklabels(["0", "1"])

for i in range(2):
    for j in range(2):
        ax2.text(
            j, i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )

plt.colorbar(im, ax=ax2)
plt.tight_layout()

fig2.savefig(os.path.join(results_dir, "macierz_pomylek_qpu.png"), dpi=300, bbox_inches="tight")
plt.show()

print(f"\nWszystkie pliki zostały pomyślnie zapisane w: {os.path.abspath(results_dir)}")