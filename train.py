import datetime

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop

from models.scan_attend_read import ScanAttendRead

checkpoint_file = 'checkpoints/scan-attend-read.h5'
batch_size = 8
epochs = 1000
learning_rate = 0.001

now = datetime.datetime.now()

input_shape = (1080, 1920)
model = ScanAttendRead(input_shape)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate),
              metrics=['accuracy'])

time = now.strftime('%Y.%m.%d %H.%M')
tensorboard = TensorBoard(log_dir='./logs/' + time)

# TODO(tom.rochette@coreteks.org): Prepare x_train/y_train/x_test/y_test
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=5,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(checkpoint_file, verbose=1, save_best_only=True),
        tensorboard
    ]
)
