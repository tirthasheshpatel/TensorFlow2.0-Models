import tkinter as tk

root= tk.Tk()

canvas = tk.Canvas(root, width = 600, height = 400,  relief = 'raised')
canvas.pack()

label1 = tk.Label(root, text='Sentiment Analysis')
label1.config(font=('helvetica', 18))
canvas.create_window(300, 35, window=label1)

label2 = tk.Label(root, text='Enter review :')
label2.config(font=('helvetica', 14))
canvas.create_window(300, 120, window=label2)


"-------------Here the review will be entered--------------"
entry = tk.Entry (root,width=70) 
canvas.create_window(300, 180, window=entry)

def sentiment():
    
    x1 = entry.get()
    
    label3 = tk.Label(root, text= 'Result is is:',font=('helvetica', 14))
    canvas.create_window(300, 270, window=label3)
    
    label4 = tk.Label(root, text= "write function which will return the result of entered review",font=('helvetica', 14, 'bold'))
    canvas.create_window(300, 300, window=label4)
    
button1 = tk.Button(text='Get the RESULT', command=sentiment, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
canvas.create_window(300, 230, window=button1)

label2 = tk.Label(root, text='by Tirth Ashesh Patel (18bce243) , Tirth Hihoriya (18bce244)')
label2.config(font=('helvetica', 14))
canvas.create_window(300, 390, window=label2)


root.mainloop()