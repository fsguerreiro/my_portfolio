import PySimpleGUI as sg
import string
import re
from collections import Counter

layout = [[sg.Text('Insert text below:', size=(40,1)), sg.Text('Result:', size=(20, 1), key='resultado')],
          [sg.MLine(size=(40, 15), key='texto'), sg.Output(size=(20, 15))],
          [sg.Button('COUNT'), sg.Radio('Ascending', group_id='r', default=True, key='asc'), sg.Radio('Descending', group_id='r', key='des'), sg.Radio('Least commom', group_id='r', key='lea'), sg.Radio('Most common', group_id='r', key='mos')]]

window = sg.Window('Word count', layout)

sg.ChangeLookAndFeel('Reddit')

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event == 'COUNT':
        tex = values['texto'].lower()
        tex = re.sub(r'[' + string.punctuation + ']', ' ', tex)
        tex = tex.split()
        y = dict(Counter(tex).most_common())
        if values['asc'] is True:
            y = sorted(y.items())
        elif values['des'] is True:
            y = sorted(y.items(), reverse=True)
        elif values['lea'] is True:
            y = sorted(y.items(), key=lambda x: x[1])
        elif values['mos'] is True:
            y = sorted(y.items(), key=lambda x: x[1], reverse=True)

        window.Element('resultado').Update(f'No. of words: {len(tex)}')
        #print('\nNo. of words:', len(tex))
        for i, j in y:
            print(str(i) + ': ' + str(j))
        print()
window.close()
