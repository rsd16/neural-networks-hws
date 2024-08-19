# Project Loneliness, done by Alireza Rashidi Laleh
# creating dataset by the help of tkinter and arrays
# python 3.7.8


from tkinter import *
import numpy as np


class create:
    def __init__(self):
        self.buttons = []
        self.button_number = 0
        self.button_values = [-1] * 26
        self.button_x = 0
        self.button_o = 0


    def change(self, button):
        if self.buttons[self.buttons.index(button)]['bg'] == 'white':
            self.buttons[self.buttons.index(button)].config(bg='red')
            self.button_values[self.buttons.index(button)] = 1
            #print(button_values[buttons.index(button)])
            #print(button_values)
        else:
            self.buttons[self.buttons.index(button)].config(bg='white')
            self.button_values[self.buttons.index(button)] = -1
            #print(button_values[buttons.index(button)])
            #print(button_values)


    # Label X == 1; Label O == 0.
    def labels(self, button):
        if button['bg'] == 'white':
            if button['text'] == 'Label X':
                self.button_x = 1
                button.config(bg='dodger blue')
                self.button_values[-1] = 1
            elif button['text'] == 'Label O':
                self.button_o = -1
                button.config(bg='spring green')
                self.button_values[-1] = -1
                
            print(self.button_values[-1])
        else:
            if button['bg'] == 'dodger blue':
                button.config(bg='white')
                if self.button_x == 1 and self.button_o == 0:
                    self.button_values[-1] = 0
                    self.button_x = 0
                elif self.button_x == 1 and self.button_o == -1:
                    self.button_x = 0
                    self.button_values[-1] = -1

            if button['bg'] == 'spring green':
                button.config(bg='white')
                if self.button_o == -1 and self.button_x == 0:
                    self.button_values[-1] = 0
                    self.button_o = 0
                elif self.button_o == -1 and self.button_x == 1:
                    self.button_o = 0
                    self.button_values[-1] = 1

            print(self.button_values[-1])


    def finalize(self, button):
        print(self.button_values)
        with open('data.txt', 'a') as file:
            for item in self.button_values:
                file.write(f'{str(item)},')

            file.write('\n')

        self.reset(button)


    def reset(self, button):
        for i in range(0, 25):
            self.button_values[i] = -1

        print(self.button_values)
        for button in self.buttons:
           button.config(bg='white')


    def program(self):
        root = Tk()
        root.title('Create Data in a Fun Way!')
        root.resizable(0, 0)
        frame = Frame(root)
        frame.pack()

        for x in range(0, 5):
            for y in range(0, 5):
                self.button_number += 1
                button = Button(frame, bg='white', height=5, width=10, text=f'{self.button_number}')
                button.grid(column=y, row=x, sticky=N+S+E+W)
                button['command'] = lambda button=button: self.change(button)
                self.buttons.append(button)

        button = Button(frame, bg='white', height=5, width=10, text='Label X')
        button.grid(column=0, row=35)
        button['command'] = lambda button=button: self.labels(button)

        button = Button(frame, bg='white', height=5, width=10, text='Label O')
        button.grid(column=1, row=35)
        button['command'] = lambda button=button: self.labels(button)

        button = Button(frame, bg='white', height=5, width=10, text='Finalize')
        button.grid(column=3, row=35)
        button['command'] = lambda button=button: self.finalize(button)

        button = Button(frame, bg='white', height=5, width=10, text='Reset')
        button.grid(column=4, row=35)
        button['command'] = lambda button=button: self.reset(button)

        root.mainloop()


user = create()
user.program()
