import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from pathlib import Path
from tkinter import *
import os, sys


def check_data():
    data=[]
    param=[]
    out=[]
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if file.endswith(".data"):
            data.append(file[:-5])
        if file.endswith(".params"):
            param.append(file[:-7])
        if file.endswith(".outs"):
            out.append(file[:-5])
    a=set(data)
    b=set(param)
    c=set(out)
    res=a.intersection(b,c)
    return list(res)


def normalize(input):
    min=np.amin(input)
    output=input-min
    max=np.amax(output)
    return output/max


def to_classes(arr,out_size):
    res=np.array([])
    arr=arr.astype(np.int)
    for val in arr:
        vec=np.zeros(out_size)
        vec[val]=1
        res=np.append(res,vec)
    return res


def read_data(name,out_size):
    name+='.data'
    data_source = open(name, 'r')
    lines = data_source.read().split('\n')
    dataset = np.array([])
    expected = np.array([])
    for line in lines:
        if not any('?' in s for s in line):
            values = line.split(',')
            expected = np.append(expected, values[-1])
            dataset = np.append(dataset, values[:-1])
    return dataset.astype(np.float), to_classes(normalize(expected[:-1].astype(np.float))*(out_size-1),out_size)


def read_params(name):
    name+='.params'
    data_source = open(name, 'r')
    return list(filter(None,data_source.read().split('\n')))


def read_outs(name):
    name+='.outs'
    data_source = open(name, 'r')
    return list(filter(None,data_source.read().split('\n')))


def train_network(datasource,in_size, out_size):
    x_train, y_train = read_data(datasource,out_size)
    x_train = np.reshape(x_train, (-1, in_size))
    y_train = np.reshape(y_train, (-1, out_size))

    model = Sequential()

    model.add(Dense(64, activation='relu', input_dim=in_size))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(out_size, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2, validation_split=0.1)
    model.save(datasource+'.hdf5')
    return model


def get_model(datasource="source.txt",in_size=1,out_size=2):
    path=datasource+'.hdf5'
    my_file = Path(path)
    if my_file.is_file():
        return load_model(path)
    else:
        return train_network(datasource,in_size,out_size)


class interface:
    labels = []
    textboxes = []
    params=[]
    def __init__(self,model, par, outs, root):
        self.params=par
        self.outs=outs
        self.root=root
        self.model=model

        vcmd = (root.register(self.validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        topFrame = Frame(root)
        topFrame.grid(row=1)
        bottomFrame = Frame(root)
        bottomFrame.grid(row=2)

        for i in range(0, len(self.params)):
            self.labels.append(Label(topFrame, text=self.params[i]))
            self.labels[i].grid(row=i, sticky=W)
            self.textboxes.append(Entry(topFrame, validate='key', validatecommand=vcmd))
            self.textboxes[i].grid(row=i, column=1)
            self.textboxes[i].insert(0, "1")

        self.resultlabel = Label(bottomFrame, text="Fill in values and press the button above.")
        self.resultlabel.grid(row=2)

        self.calcButton = Button(bottomFrame, text="evaluate", command=self.respond)
        self.calcButton.grid(row=1)
        self.reset = Button(root, text="Change model", command=self.restart_program).grid(row=4)

    def validate(self, action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
        if action == '1':
            if text in '0123456789.-':
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    return False
            else:
                return False
        else:
            return True


    def respond(self):
        values=np.array([])
        for i in range(0, len(self.params)):
            if self.textboxes[i].get() == "":
                self.textboxes[i].insert(0, "1")
            values = np.append(values, self.textboxes[i].get())
        values = values.astype(np.float)
        values = values.reshape(-1,len(values))
        result = self.model.predict(values)[0]
        resp=np.argmax(result)
        self.resultlabel['text'] = self.outs[resp] + ' with {0:.2f} % probability'.format(result[resp] * 100)

    def restart_program(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

class model_picker:


    def __init__(self,root):
        self.picked=StringVar(root)
        self.detected=check_data()
        self.picked.set(self.detected[0])
        self.text=Label(root, text='Select data source, make sure there are .data, .outs and .params files in the main directory, as described in readme file.')
        self.text.grid(row=1)
        self.drop=OptionMenu(root, self.picked, *self.detected)
        self.drop.grid(row=2)
        self.start=Button(root, text='Select', command=self.launch)
        self.start.grid(row=3)
    def launch(self):
        params = read_params(self.picked.get())
        params[-1]=params[-1][:-1]
        outs = read_outs(self.picked.get())
        outs = sorted(outs, key=str.lower)
        model = get_model(self.picked.get(),len(params),len(outs))
        self.text.destroy()
        self.start.destroy()
        self.drop.destroy()
        selected = interface(model, params, outs, root)


root = Tk()


selector = model_picker(root)

root.mainloop()