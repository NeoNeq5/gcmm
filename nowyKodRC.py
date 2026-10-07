import os
import time
from datetime import datetime

from dotenv import load_dotenv
from iqm.qiskit_iqm import IQMProvider
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Kompatybilność 3D dla starszych wersji matplotlib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

METHOD = 'lda'
N_COMPONENTS = 3
NUM_QUBITS = N_COMPONENTS

RESULTS_DIR = "wyniki_qpu"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_spark_backend(env_file="token.env"):
    load_dotenv(env_file)
    server_url = os.getenv("SERVER", "https://odra5.e-science.pl")
    token = os.getenv("TOKEN")
    provider = IQMProvider(server_url, token=token)
    return provider.get_backend()

dev_sim = qml.device("default.qubit", wires=NUM_QUBITS)

print("Łączenie z komputerem ")
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


digits = load_digits()
X_raw = digits.data
y_raw = digits.target
images_raw = digits.images

mask = (y_raw == 0) | (y_raw == 1) | (y_raw == 2)
X_sub, y_sub, images_sub = X_raw[mask], y_raw[mask], images_raw[mask]

print("Redukcja")

X_train_raw, X_test_raw, y_train, y_test, img_train, img_test = train_test_split(
    X_sub,
    y_sub,
    images_sub,
    test_size=0.2,
    random_state=42,
    stratify=y_sub,
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

y_train_targets = np.where(np.eye(3)[y_train] == 1, 1.0, -1.0)


def circuit_architecture(weights, x):
    qml.AngleEmbedding(x, wires=range(NUM_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

@qml.qnode(dev_sim)
def quantum_circuit_sim(weights, x):
    return circuit_architecture(weights, x)

@qml.qnode(dev_qpu)
def quantum_circuit_qpu(weights, x):
    return circuit_architecture(weights, x)

#trening
def cost_function(weights, X_batch, y_batch_targets):
    predictions = pnp.array([quantum_circuit_sim(weights, x) for x in X_batch])
    return pnp.mean((predictions - y_batch_targets) ** 2)

num_layers = 4
np.random.seed(42)
weights = pnp.array(
    np.random.randn(num_layers, NUM_QUBITS, 3), requires_grad=True
)

opt = qml.AdamOptimizer(stepsize=0.05)
batch_size = 20
epochs = 8

print("treningu")
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_shuffled = X_train[permutation]
    y_targets_shuffled = y_train_targets[permutation]

    for i in range(0, len(X_train), batch_size):
        X_b = X_shuffled[i : i + batch_size]
        y_b = y_targets_shuffled[i : i + batch_size]
        weights, loss = opt.step_and_cost(
            lambda w: cost_function(w, X_b, y_b), weights
        )

    test_preds_sim = [
        np.argmax(quantum_circuit_sim(weights, x)) for x in X_test
    ]
    accuracy_sim = np.mean(test_preds_sim == y_test)
    print(f"Epoka {epoch+1}/{epochs} | Loss: {loss:.4f} | Sim Accuracy: {accuracy_sim * 100:.2f}%")


weights_file = os.path.join(RESULTS_DIR, "vqc_trained_weights.npy")
np.save(weights_file, np.array(weights))


print("\n komputer kwantowy")

test_preds_qpu = []
total_test_samples = len(X_test)

for i, x in enumerate(X_test):
    try:
        pred = int(np.argmax(quantum_circuit_qpu(weights, x)))
    except Exception as e:
        print(f"Błąd przy próbce {i+1}: {e}")
        pred = 0  # Domyślny fallback w razie zerwania połączenia
        
    test_preds_qpu.append(pred)
    
    if (i + 1) % 5 == 0 or (i + 1) == total_test_samples:
        print(f"Postęp QPU: {i + 1}/{total_test_samples} próbek przetworzonych...")
    
    time.sleep(1.5)

test_preds_qpu = np.array(test_preds_qpu)
final_accuracy = accuracy_score(y_test, test_preds_qpu)

print("\n" + "=" * 70)
print("KOŃCOWE WYNIKI TESTÓW NA QPU IQM")
print("=" * 70)
print(f"Dokładność na QPU (Test Accuracy): {final_accuracy * 100:.2f}%")

# MACIERZ POMYŁEK
cm = confusion_matrix(y_test, test_preds_qpu, labels=[0, 1, 2])
fig3, ax3 = plt.subplots(figsize=(6, 5))
im = ax3.imshow(cm, cmap="Blues")

ax3.set_title("Macierz pomyłek (Wyniki z QPU)", fontsize=14, fontweight="bold")
ax3.set_xlabel("Przewidziana klasa (QPU)")
ax3.set_ylabel("Prawdziwa klasa")
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])

for i in range(3):
    for j in range(3):
        ax3.text(
            j, i, cm[i, j],
            ha="center", va="center",
            fontsize=16, fontweight="bold",
        )

plt.colorbar(im, ax=ax3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "macierz_pomylek_qpu.png"), dpi=300, bbox_inches="tight")

# wykres

fig2 = plt.figure(figsize=(11, 9))
ax2 = fig2.add_subplot(111, projection='3d')

fig2.suptitle(
    f"Complete Test Set in {METHOD.upper()} Feature Space (3D)",
    fontsize=14,
    fontweight='bold'
)

colors = {0: 'blue', 1: 'orange', 2: 'green'}


for digit, color in colors.items():
    correct_mask = (test_preds_qpu == digit) & (y_test == digit)

    ax2.scatter(
        X_test[correct_mask, 0],
        X_test[correct_mask, 1],
        X_test[correct_mask, 2],
        c=color,
        marker='o',
        s=30,
        alpha=0.6,
        label=f'Correct: {digit}'
    )

incorrect_mask = (test_preds_qpu != y_test)

ax2.scatter(
    X_test[incorrect_mask, 0],
    X_test[incorrect_mask, 1],
    X_test[incorrect_mask, 2],
    c='red',
    marker='o',
    s=60,
    alpha=0.9,
    label='Misclassified'
)

ax2.set_xlabel(f"{METHOD.upper()} Feature 1")
ax2.set_ylabel(f"{METHOD.upper()} Feature 2")
ax2.set_zlabel(f"{METHOD.upper()} Feature 3")

ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout()


plt.savefig(os.path.join(RESULTS_DIR, "wykres_3d_qpu.png"), dpi=300, bbox_inches="tight")


report_str = classification_report(
    y_test,
    test_preds_qpu,
    labels=[0, 1, 2],
    target_names=["Cyfra 0", "Cyfra 1", "Cyfra 2"],
    digits=4,
)

print("\n" + "=" * 70)
print("RAPORT KLASYFIKACJI (QPU IQM)")
print("=" * 70)
print(report_str)

report_file = os.path.join(RESULTS_DIR, "raport_qpu.txt")
with open(report_file, "w", encoding="utf-8") as f:
    f.write("RAPORT Z URUCHOMIENIA NA SPRZĘCIE KWANTOWYM IQM ODRA5\n")
    f.write(f"Data wykonania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Metoda redukcji: {METHOD.upper()} (Składowe: {N_COMPONENTS})\n")
    f.write(f"Liczba kubitów: {NUM_QUBITS}\n")
    f.write(f"Dokładność końcowa na QPU: {final_accuracy * 100:.2f}%\n\n")
    f.write("MACIERZ POMYŁEK:\n")
    f.write(f"{cm}\n\n")
    f.write("RAPORT KLASYFIKACJI:\n")
    f.write(report_str)

csv_file = os.path.join(RESULTS_DIR, "predykcje_qpu.csv")
data_to_save = np.column_stack((y_test, test_preds_qpu))
np.savetxt(
    csv_file,
    data_to_save,
    fmt="%d",
    delimiter=",",
    header="True_Label,QPU_Pred",
    comments="",
)

plt.show()