import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential

class MODEL:
    def __init__(self) -> None:
        self.numFeatures =129 
        self.numSegments =8 

    def build_model(self, l2_strength):
        inputs = Input(shape=[self.numFeatures,self.numSegments,1])
        x = inputs

        # -----
        x = tf.keras.layers.ZeroPadding2D(((4,4), (0,0)))(x)
        x = Conv2D(filters=18, kernel_size=[9,8], strides=[1, 1], padding='valid', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        skip0 = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(skip0)
        x = BatchNormalization()(x)

        x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        # -----
        x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        skip1 = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(skip1)
        x = BatchNormalization()(x)

        x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        # ----
        x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        # ----
        x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = x + skip1
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        # ----
        x = Conv2D(filters=18, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=30, kernel_size=[5,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = x + skip0
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=8, kernel_size=[9,1], strides=[1, 1], padding='same', use_bias=False,
                    kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        # ----
        x = tf.keras.layers.SpatialDropout2D(0.2)(x)
        x = Conv2D(filters=1, kernel_size=[129,1], strides=[1, 1], padding='same')(x)

        model = Model(inputs=inputs, outputs=x)

        optimizer = tf.keras.optimizers.Adam(3e-4)
        #optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=3e-4)

        model.compile(optimizer=optimizer, loss='mse', 
                        metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])
        return model
    
    def model_evaluate(self,model,test_data):
        baseline_val_loss = model.evaluate(test_data)[0]
        print(f"Baseline accuracy {baseline_val_loss}")
        return baseline_val_loss
    
    def model_save(self, model,test_data,baseline_val_loss):
        val_loss = model.evaluate(test_data)[0]
        if val_loss < baseline_val_loss:
                print("New model saved.")
        model.save('model.h5')
