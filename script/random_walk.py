import random
import matplotlib.pyplot as plt

def random_walk_2d(steps):
  x, y = 0, 0
  x_values, y_values = [x], [y]

  for _ in range(steps):
    step = random.choice(['up', 'down', 'left', 'right'])
    if step == 'up':
      y += 1
    elif step == 'down':
      y -= 1
    elif step == 'left':
      x -= 1
    elif step == 'right':
      x += 1

    x_values.append(x)
    y_values.append(y)

  return x_values, y_values

def random_walk_1d(steps):
  x = 0
  x_values = [x]

  for _ in range(steps):
    step = random.choice(['left', 'right'])
    if step == 'left':
      x -= 1
    elif step == 'right':
      x += 1

    x_values.append(x)

  return x_values

def plot_walk(x_values, y_values):
  plt.figure(figsize=(10, 10))
  plt.plot(x_values, y_values, marker='o')
  plt.title('2D Random Walk')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  steps = 1000
  x_values, y_values = random_walk_2d(steps)
  plot_walk(x_values, y_values)