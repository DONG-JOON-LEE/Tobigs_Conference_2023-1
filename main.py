import os
import tensorflow as tf
import datetime
from Tobigs.Data.dataset import Dataset
from Tobigs.Model.sound_parser import SoundParser
from Tobigs.Model.model import MODEL

def main():
    clean_train_file = sorted(os.listdir('./Clean_audio_train/'))
    clean_val_file = sorted(os.listdir('./Clean_audio_val/'))
    noise_file = sorted(os.listdir('./noise/'))

    config = {'windowLength': 256,
            'overlap': round(0.25 * 256),
            'fs': 22100,
            'audio_max_duration': 0.8}

    train_dataset = Dataset(clean_train_file, noise_file, train=True, **config)
    train_dataset.create_tf_record(prefix='train', subset_size=1)

    val_dataset = Dataset(clean_val_file, noise_file, train=False, **config)
    val_dataset.create_tf_record(prefix='val', subset_size=1)

    sound_parser = SoundParser()
    train_data, test_data = sound_parser.get_data()


    model = MODEL()
    model_train = model.build_model(l2_strength=0.00001)
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, baseline=None)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./denoiser_cnn_log_mel_generator.h5', 
                                                         monitor='val_loss', save_best_only=True)

    model_train.fit(train_data,
         steps_per_epoch=32, # you might need to change this
         validation_data=test_data,
         epochs=400,
         callbacks=[early_stopping_callback, tensorboard_callback, checkpoint_callback]
        )
    
    baseline_val_loss=model.model_evaluate(model_train, test_data)
    model.model_save(model_train, test_data, baseline_val_loss)


if __name__ == "__main__": 
    main()
