import matplotlib.pyplot as plt

def plot_mIoU_validation(N_epoch, mIoU_list):

    print(f'x = {N_epoch}, y = {len(mIoU_list)}')

    plt.title(f'mean IoU')
    plt.plot(list(range(N_epoch)), mIoU_list)
