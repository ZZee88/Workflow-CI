import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow

def train_ci_model():
    # Mengaktifkan autolog bawaan proyek MLflow
    mlflow.tensorflow.autolog()

    # Path dataset diatur searah dengan folder repository GitHub CI
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PREPROCESSED_DIR = os.path.join(CURRENT_DIR, "dataset_preprocessing")

    TRAIN_DIR = os.path.join(PREPROCESSED_DIR, "train")
    VAL_DIR = os.path.join(PREPROCESSED_DIR, "val")

    # Loader data sederhana (Rescale 0-1)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Menggunakan generator dummy/minimalis untuk kebutuhan simulasi CI otomatis
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(150, 150), batch_size=2, class_mode='binary'
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(150, 150), batch_size=2, class_mode='binary'
    )

    # Arsitektur Wajib: Sequential, Conv2D, Pooling
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Eksekusi ringkas 1 epoch untuk syarat trigger re-training CI
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model.fit(train_generator, epochs=1, validation_data=val_generator)
        
        # Menyimpan model lokal untuk kebutuhan build docker image di langkah berikutnya
        model.save("ci_model.keras")

        # Catat Run ID langsung di dalam block saat run masih aktif
        run_id = mlflow.active_run().info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

if __name__ == "__main__":
    train_ci_model()
