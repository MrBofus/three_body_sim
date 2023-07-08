import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math



def algorithms(y, flag, k):
    if flag == 0:
        title = 'bubble sort algorithm'
        k = 0
        
        T = 0
        for i in range(len(y)-1):
            
            if y[i] > y[i+1]:
                y[i], y[i+1] = y[i+1], y[i]
                
            elif y[i] < y[i+1]:
                T += 1
    
    elif flag == 1:
        title = 'insertion sort algorithm'
        
        T = 0
        key_item = y[k]
        
        j = k - 1
        
        while j >= 0 and y[j] > key_item:
            y[j+1] = y[j]
            j -= 1
        
        y[j+1] = key_item
        
        for i in range(1, len(y)):
            if y[i-1] < y[i]:
                T += 1
        k += 1
        
    
    elif flag == 2:
        title = 'quicksort algorithm'
        k = 0
        
        T = 0
        
        low, same, high = [], [], []
        pivot = y[random.randint(0, len(y) - 1)]
        
        for item in y:
            if item < pivot:
                low.append(item)
            elif item == pivot:
                same.append(item)
            elif item > pivot:
                high.append(item)
        
        y = low + same + high
        
        for i in range(1, len(y)):
            if y[i-1] < y[i]:
                T += 1
    
    elif flag == 3:
        title = 'selection sort algorithm'
        
        T = 0
        
        min_idx = k
        for i in range(k+1, len(y)):
            
            if y[i] < y[min_idx]:
                min_idx = i
        
        y[k], y[min_idx] = y[min_idx], y[k]
        
        for i in range(1, len(y)):
            if y[i-1] < y[i]:
                T += 1
        
        k += 1
    
    elif flag == 4:
        title = 'shell sort algorithm'
        
        T = 0
        
        n = len(y)
        if k == 0: k = int(math.log2(n))
        interval = 2**k - 1
        
        for i in range(interval, n):
            temp = y[i]
            j = i
            while j >= interval and y[j - interval] > temp:
                y[j] = y[j - interval]
                j -= interval
            y[j] = temp
            
        k -= 1
        
        for i in range(1, len(y)):
            if y[i-1] < y[i]:
                T += 1
                
    return y, T, title, k



fig, ax = plt.subplots(facecolor='black') 
plt.rcParams['toolbar'] = 'None' # Remove tool bar (upper bar)
fig.canvas.window().statusBar().setVisible(False) # Remove status bar (bottom bar)

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

counter = 0
while True:
    
    ax.clear()
    
    x = [0]*100
    y = [0]*100
    for i in range(len(y)):
        x[i] = i
        y[i] = i
    
    random.shuffle(y)
    
    ax.set_facecolor("black")
    ax.bar(x, y, color='white')
    plt.pause(2)
    
    k_old = 0
    while True:
        ax.clear()
        
        
        y, T, title, k = algorithms(y, counter%5, k_old)
        k_old = k
        
        ax.bar(x, y, color='white')
        ax.set_facecolor("black")
        ax.set_title(title, color='white')
        
        k += 1
        plt.pause(0.01)
        
        if T == len(y)-1:
            break
    
    counter += 1
    time.sleep(2)