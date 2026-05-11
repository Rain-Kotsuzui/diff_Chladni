import warp as wp
import numpy as np
import matplotlib.pyplot as plt
from parameter import PlateParams
from direct_solver import DifferentiableDirectSolver
from device_manager import WARP_DEVICE, init_devices

init_devices()

def getDepth(N: int,L:float):
    
    h_np = np.ones(N * N, dtype=np.float32) * 0.001
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            x = (j*1.0 / N)
            y = (i*1.0 / N)
            # if(y<x*1.9 and y<(1-x)*1.9 and y>0.1):
            #     h_np[idx] = 0.001
            # if( (x-0.2)**2+(y-0.5)**2 < 0.11**2 or (x-0.8)**2+(y-0.5)**2 < 0.11**2 or (x-0.5)**2+(y-0.2)**2 < 0.11**2 or (x-0.5)**2+(y-0.8)**2 < 0.11**2):
            # if( x>0.7 and x<0.8):
                #  h_np[idx] = 0.000
    return h_np

def main():
    N = 128       
    L = 0.3         # 30cm
    
    p = PlateParams()
    p.E = 70.0e9     
    p.nu = 0.33      
    p.rho = 2700.0   
    p.dx = L / N     
    p.dy = L / N     
    p.eta = 1e-4    # 阻尼

    chladni_solver = DifferentiableDirectSolver(p, N, N)

    start_f = 500.0   
    end_f = 2000.0
    steps = 200      
    freqs = np.linspace(start_f, end_f, steps)

    
    h_np = getDepth(N,L)
    h_wp = wp.from_numpy(h_np, device=WARP_DEVICE)
    h_np = h_np.reshape((N, N)) * 1000.0 

    # 力
    fr_np = np.zeros(N * N, dtype=np.float32)
    
    # 第一个源：强度 1.0
    idx1 = (N // 2) * N + (N // 2)
    fr_np[idx1] = 1.0 
    
    # # 第二个源：强度 10.0
    # idx2 = (3 * N // 4) * N + (N // 4)
    # fr_np[idx2] = 1.0 
    
    
    # idx2 = (3 * N // 4) * N + (3*N // 4)
    # fr_np[idx2] = 1.0 


    # 绘图
    fr = wp.from_numpy(fr_np, device=WARP_DEVICE)
    fi = wp.zeros(N * N, dtype=float, device=WARP_DEVICE)

    source_indices = np.where(fr_np > 0)[0]
    source_rows = source_indices // N
    source_cols = source_indices % N
    source_x_cm = source_cols * (L / N) * 100.0
    source_y_cm = source_rows * (L / N) * 100.0
    source_mags = fr_np[source_indices]

    plt.ion() 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7), facecolor='black')

    dummy_data = np.zeros((N, N))
    im1 = ax1.imshow(dummy_data, cmap='magma', origin='lower', extent=[0, L*100, 0, L*100], vmin=0, vmax=1)
    im2 = ax2.imshow(dummy_data, cmap='viridis', origin='lower', extent=[0, L*100, 0, L*100])
    
    x_coords = np.linspace(0, L*100, N)
    y_coords = np.linspace(0, L*100, N)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    im3 = ax3.imshow(h_np, cmap='plasma', origin='lower', extent=[0, L*100, 0, L*100])
    ax3.set_title("Plate Thickness (mm)", color='white')
    cbar = fig.colorbar(im3, ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Thickness (mm)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    for ax in [ax1, ax2,ax3]:
        ax.scatter(source_x_cm, source_y_cm, 
                   s=source_mags * 20 + 50,  
                   c='cyan', marker='x', 
                   linewidths=2, zorder=5, label='Sources')
        
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.set_facecolor('black')

    ax1.set_title("Chladni Pattern (Nodes)", color='white')
    ax2.set_title("energy field", color='white')


    
    for f in freqs:
        p.omega = 2.0 * np.pi * f
        try:
            w_complex = chladni_solver.solve(h_wp, fr, fi)
            wr, wi = w_complex.real, w_complex.imag

            amp = np.sqrt(wr**2 + wi**2).reshape((N, N))
            
            clip_max = np.percentile(amp, 98)
            if clip_max < 1e-20: clip_max = 1e-20
            
            display_pattern = np.exp(-(amp / (clip_max * 0.15))**2)
            plt.imsave('pattern.png', display_pattern, cmap='viridis')

            im1.set_data(display_pattern)
            
            
            energy_field = (wr**2.0 + wi**2.0).reshape( N, N)
            im2.set_data(energy_field.reshape(N, N))
            for c in ax2.collections:
                    c.remove()
            
            non_zero_energy = energy_field[energy_field > 1e-18]
            if non_zero_energy.size > 0:
                threshold = np.percentile(non_zero_energy, 5)
                ax2.contour(X_grid, Y_grid, energy_field, levels=[threshold], colors='white', linewidths=1.5)
            im2.set_clim(np.min(energy_field), np.max(energy_field))
                

            
            ax1.set_title(f"Frequency: {f:.1f} Hz", color='white', fontsize=16)
            
            plt.draw()
            plt.pause(0.01)
            
        except Exception as e:
            print(f"频率 {f:.1f}Hz 求解出错: {e}")
            continue

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()