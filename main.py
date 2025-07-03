import numpy as np
import matplotlib.pyplot as plt

# Debug flags
DEBUG_PRINT_FIELDS = True   # Print field values at each step
DEBUG_PAUSE = False          # Pause after each step for debugging
DEBUG_STEP = 10              # Print every N steps if DEBUG_PRINT_FIELDS is True

# Stałe fizyczne
c0 = 299_792_458        # prędkość światła [m/s]
eps0 = 8.854e-12        # przenikalność próżni [F/m]
mu0 = 4*np.pi*1e-7      # przenikalność magnetyczna próżni [H/m]

# Parametry siatki i czasu
Nx, Ny = 200, 200
dx = dy = 1e-3          # krok przestrzenny [m]
dt = 0.5 * min(dx, dy) / c0  # krok czasowy [s]
n_steps = 1000

# Tablice pól
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# Współczynniki aktualizacji
ceze = 1.0
cezh = dt / (eps0 * dx)
chxh = 1.0
chxe = dt / (mu0 * dy)

# Proste warunki brzegowe Mur
Ez_x0, Ez_x1 = np.copy(Ez[1, :]), np.copy(Ez[-2, :])
Ez_y0, Ez_y1 = np.copy(Ez[:, 1]), np.copy(Ez[:, -2])

# Pętla czasowa
for n in range(1, n_steps+1):
    # Aktualizacja Hx, Hy
    Hx[:, :-1] += chxe * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:-1, :] -= chxh * (Ez[1:, :] - Ez[:-1, :])

    # Aktualizacja Ez
    Ez[1:, 1:] += cezh * ((Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))

    # Warunki brzegowe Mur
    Ez[0, :] = Ez_x0 + (c0*dt - dx)/(c0*dt + dx) * (Ez[1, :] - Ez[0, :])
    Ez[-1, :] = Ez_x1 + (c0*dt - dx)/(c0*dt + dx) * (Ez[-2, :] - Ez[-1, :])
    Ez[:, 0] = Ez_y0 + (c0*dt - dy)/(c0*dt + dy) * (Ez[:, 1] - Ez[:, 0])
    Ez[:, -1] = Ez_y1 + (c0*dt - dy)/(c0*dt + dy) * (Ez[:, -2] - Ez[:, -1])

    Ez_x0, Ez_x1 = np.copy(Ez[1, :]), np.copy(Ez[-2, :])
    Ez_y0, Ez_y1 = np.copy(Ez[:, 1]), np.copy(Ez[:, -2])

    # Punktowe źródło sinusoidalne w centrum
    Ez[Nx//2, Ny//2] += np.sin(2*np.pi*1e9 * n * dt)

    # Debug output
    if DEBUG_PRINT_FIELDS and n % DEBUG_STEP == 0:
        print(f"Step {n}: Ez center = {Ez[Nx//2, Ny//2]:.4e}, max = {Ez.max():.4e}, min = {Ez.min():.4e}")
        if DEBUG_PAUSE:
            input("Press Enter to continue...")

    # Wizualizacja co 50 kroków
    if n % 50 == 0:
        plt.clf()
        plt.imshow(Ez.T, cmap='RdBu', origin='lower')
        plt.title(f'Krok czasowy: {n}')
        plt.pause(0.01)

plt.show()
