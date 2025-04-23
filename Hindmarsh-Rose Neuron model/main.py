import numpy as np
import matplotlib.pyplot as plt

# Model parametreleri (Ödevde verilen değerler)
a = 1.0
b = 3.0
c = 1.0
d = 5.0
r = 0.006 
s = 4.0
x_bar = -1.6
I0 = 3.0    # Harici stimülasyon akımı

# Başlangıç koşulları
x0 = 0.005
y0 = 0.003
z0 = 0.003

# Zaman ayarları
Ts = 0.1    # Örnekleme periyodu (saniye)
N = 1000    # Adım sayısı
t = np.linspace(0, N*Ts, N)  # Zaman vektörü (0-100 saniye)

# Hindmarsh-Rose model dinamikleri
def hr_model(x, y, z, I0):
    dx = y - a*x**3 + b*x**2 - z + I0
    dy = c - d*x**2 - y
    dz = r*(s*(x - x_bar) - z)
    return dx, dy, dz

# Euler integrasyonu
def euler_integrate():
    x, y, z = np.zeros(N), np.zeros(N), np.zeros(N)
    x[0], y[0], z[0] = x0, y0, z0
    
    for i in range(N-1):
        dx, dy, dz = hr_model(x[i], y[i], z[i], I0)
        x[i+1] = x[i] + Ts * dx
        y[i+1] = y[i] + Ts * dy
        z[i+1] = z[i] + Ts * dz
    
    return x, y, z

# Runge-Kutta 4. derece (RK4) integrasyonu
def rk4_integrate():
    x, y, z = np.zeros(N), np.zeros(N), np.zeros(N)
    x[0], y[0], z[0] = x0, y0, z0
    
    for i in range(N-1):
        # k1 adımı
        k1x, k1y, k1z = hr_model(x[i], y[i], z[i], I0)
        
        # k2 adımı
        k2x, k2y, k2z = hr_model(x[i] + Ts/2*k1x, 
                                y[i] + Ts/2*k1y, 
                                z[i] + Ts/2*k1z, I0)
        
        # k3 adımı
        k3x, k3y, k3z = hr_model(x[i] + Ts/2*k2x, 
                                y[i] + Ts/2*k2y, 
                                z[i] + Ts/2*k2z, I0)
        
        # k4 adımı
        k4x, k4y, k4z = hr_model(x[i] + Ts*k3x, 
                                y[i] + Ts*k3y, 
                                z[i] + Ts*k3z, I0)
        
        # Toplam ağırlıklı ortalama
        x[i+1] = x[i] + (Ts/6)*(k1x + 2*k2x + 2*k3x + k4x)
        y[i+1] = y[i] + (Ts/6)*(k1y + 2*k2y + 2*k3y + k4y)
        z[i+1] = z[i] + (Ts/6)*(k1z + 2*k2z + 2*k3z + k4z)
    
    return x, y, z

# Simülasyonları çalıştır
x_euler, y_euler, z_euler = euler_integrate()
x_rk4, y_rk4, z_rk4 = rk4_integrate()

# Grafikleri oluştur
plt.figure(figsize=(14, 10))

# Euler sonuçları
plt.subplot(2, 2, 1)
plt.plot(t, x_euler, 'b', label='x (membrane potential)')
plt.plot(t, y_euler, 'r', label='y (fast current)')
plt.title('Euler Integration - X and Y')
plt.xlabel('Time [sec]')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.ylim(-3, 3) 

plt.subplot(2, 2, 2)
plt.plot(t, z_euler, 'g', label='z (slow current)')
plt.title('Euler Integration - Z')
plt.xlabel('Time [sec]')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.ylim(-0.5, 2.5)

# RK4 sonuçları
plt.subplot(2, 2, 3)
plt.plot(t, x_rk4, 'b', label='x (membrane potential)')
plt.plot(t, y_rk4, 'r', label='y (fast current)')
plt.title('RK4 Integration - X and Y')
plt.xlabel('Time [sec]')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.ylim(-3, 3)

plt.subplot(2, 2, 4)
plt.plot(t, z_rk4, 'g', label='z (slow current)')
plt.title('RK4 Integration - Z')
plt.xlabel('Time [sec]')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.ylim(-0.5, 2.5)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_euler, y_euler, 'b')
plt.title('Euler - Faz Portresi (X vs Y)')
plt.xlabel('Membrane Potential (x)')
plt.ylabel('Fast Current (y)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_rk4, y_rk4, 'r')
plt.title('RK4 - Faz Portresi (X vs Y)')
plt.xlabel('Membrane Potential (x)')
plt.ylabel('Fast Current (y)')
plt.grid(True)

plt.tight_layout()
plt.show()