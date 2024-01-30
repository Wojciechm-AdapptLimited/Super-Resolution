from datetime import datetime
import argparse

import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from src.models import create_autoencoder, create_srcnn, create_unet

IMG_SHAPE = (400, 400)
INPUT_SHAPE = (200, 200)


def get_model(name: str):
    if name == "SRCNN":
        model = create_srcnn(IMG_SHAPE, INPUT_SHAPE)
        model.build(IMG_SHAPE + (3, ))
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss='mae',
                      metrics=['mse', 'mean_squared_logarithmic_error']
                      )
        return model

    if name == "Autoencoder":
        model = create_autoencoder(IMG_SHAPE, INPUT_SHAPE)
        model.build(IMG_SHAPE + (3, ))
        model.compile(optimizer=keras.optimizers.Adam(), 
                      loss='mae',
                      metrics=['mse', 'sum']
                      )
        return model
    if name == "UNet":
        model = create_unet(IMG_SHAPE, INPUT_SHAPE)
        model.build(IMG_SHAPE + (3, ))
        model.compile(optimizer='adam', 
                    loss='mean_absolute_error',
                    metrics=['mean_squared_error', 'mean_squared_logarithmic_error']
                    )
        return model

    raise ValueError(f"Unknown model name: {name}")



def get_generator(datagen: ImageDataGenerator, dir: str, target_size=IMG_SHAPE, batch_size=32):
    return datagen.flow_from_directory(
        dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='input'
    )


def train(model, train_generator, validation_generator, epochs=10):
    early_stopping = EarlyStopping(monitor='val_mean_squared_error', patience=3, restore_best_weights=True)

    model.fit(
            train_generator,
            epochs=epochs,
            verbose=0,
            validation_data=validation_generator,
            callbacks=[early_stopping]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="script to launch model training")

    parser.add_argument('architecture')
    parser.add_argument('-E', '--epochs', type=int, default=10)

    args = parser.parse_args()

    datagen = ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.2,
            shear_range=0.2,
            rotation_range=10,
            horizontal_flip=True,
        )

    train_generator = get_generator(datagen, "./data/kaggle/train")
    validation_generator = get_generator(datagen, "./data/kaggle/valid")

    model = get_model(args.architecture)

    train(model, train_generator, validation_generator, epochs=args.epochs)

    model.save(f"./model/{args.architecture}.keras")

