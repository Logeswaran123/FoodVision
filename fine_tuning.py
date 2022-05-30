"""
FoodVision101

Fine tuning the model saved from food_vision.py
"""
import math
import tensorflow as tf

from preprocess import preprocess
from utils import create_tensorboard_callback

SEED=42
INITIAL_LR = 0.0001
tf.random.set_seed(SEED)

def lr_step_decay(epoch, lr):
    """
    Learning rate decay function
    """
    drop_rate = 0.5
    epochs_drop = 10.0
    return INITIAL_LR * math.pow(drop_rate, math.floor(epoch/epochs_drop))

def main():
    # Load model
    saved_model_path = r"C:\Users\loges\Documents\Tensorflow Practice Dir\ComputerVision\FoodVision\foodvision101_model_3.h5"
    cnn_model = tf.keras.models.load_model(saved_model_path)
    cnn_model.summary()

    train_dir = r"C:\Users\loges\Documents\Datasets\FoodVision\archive\train_test_split\train"
    test_dir = r"C:\Users\loges\Documents\Datasets\FoodVision\archive\train_test_split\test"
    # Preprocess for model
    train_data, test_data = preprocess(train_dir=train_dir,
                                        test_dir=test_dir,
                                        label_mode="categorical",
                                        target_size=(224, 224),
                                        seed=SEED)

    # Evaluate the loaded model
    cnn_model.evaluate(test_data)

    # Since we have large dataset, set all layers as trainable (i.e. unfreeze all layers)
    for layer in cnn_model.layers:
        layer.trainable = True # set all layers to trainable
        print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

    # Create EarlyStopping callback to stop training
    # if model's val_loss doesn't improve for 3 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    patience=3)

    # Create learning rate reduction callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    factor=0.2,
                                                    patience=2,
                                                    verbose=1,
                                                    min_lr=1e-7)

    # Create ModelCheckpoint callback to save model's progress
    checkpoint_path = "model_checkpoints/FineTune/model_1.ckpt"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      montior="val_accuracy",
                                                      save_best_only=True,
                                                      verbose=1)

    # Create Learning rate scheduler callback to modify learning rate in model's training
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)

    # Create Tensorboard callback
    tensorboard_callback = create_tensorboard_callback("training_logs",
                            "efficientnetb0_101_classes_all_data_fine_tuned")

    # Compile the model
    cnn_model.compile(loss="categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
                      metrics=["accuracy"])

    # Fit the model
    cnn_model.fit(train_data,
                  epochs=100,
                  steps_per_epoch=len(train_data),
                  validation_data=test_data,
                  validation_steps=int(0.15 * len(test_data)),
                  callbacks=[tensorboard_callback,
                            model_checkpoint,
                            lr_scheduler,
                            early_stopping,
                            reduce_lr])

    # Save the model
    cnn_model.save('foodvision101_model_3_fine_tuned.h5')

    # Load the saved fine_tuned model, and evaluate on test data
    saved_fine_tuned_model_path = r"C:\Users\loges\Documents\Tensorflow Practice Dir\ComputerVision\FoodVision\foodvision101_model_3_fine_tuned.h5"
    fine_tuned_cnn_model = tf.keras.models.load_model(saved_fine_tuned_model_path)
    fine_tuned_cnn_model.summary()
    fine_tuned_cnn_model.evaluate(test_data)

if __name__ == "__main__":
    main()
