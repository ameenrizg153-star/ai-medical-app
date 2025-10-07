import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# 1. إنشاء بيانات تدريب وهمية لـ ECG
def generate_fake_ecg_data(num_samples):
    X = []
    y = []
    for i in range(num_samples):
        if i % 2 == 0:
            # إشارة طبيعية
            signal = np.sin(np.linspace(0, 4 * np.pi, 187)) + np.random.normal(0, 0.1, 187)
            X.append(signal)
            y.append(0) # 0 = normal
        else:
            # إشارة غير طبيعية
            signal = np.sin(np.linspace(0, 4 * np.pi, 187)) * np.random.uniform(0.5, 1.5, 187) + np.random.normal(0, 0.3, 187)
            X.append(signal)
            y.append(1) # 1 = abnormal
    return np.array(X), np.array(y)

X_train, y_train = generate_fake_ecg_data(100)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# 2. بناء نموذج الشبكة العصبونية
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(187, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("🧠 تم بناء نموذج تحليل ECG.")

# 3. تدريب النموذج
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
print("💪 تم تدريب النموذج.")

# 4. حفظ النموذج المدرب في ملف H5
model.save('ecg_classifier_model.h5')
print("✅ ملف 'ecg_classifier_model.h5' تم إنشاؤه بنجاح.")

# 5. إنشاء وحفظ الإشارات التجريبية للعرض
sample_signals = {
    'normal': np.sin(np.linspace(0, 4 * np.pi, 187)) + np.random.normal(0, 0.1, 187),
    'abnormal': np.sin(np.linspace(0, 4 * np.pi, 187)) * 1.5 + np.random.normal(0, 0.3, 187)
}
np.save('sample_ecg_signals.npy', sample_signals)
print("✅ ملف 'sample_ecg_signals.npy' تم إنشاؤه بنجاح.")
