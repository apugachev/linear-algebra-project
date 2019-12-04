import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from progressbar import progressbar as pb
from PIL import Image

class Picture_decomposition:
    '''
    Класс для работы с изображениями
    '''
    
    def __init__(self, path_name, pics_name, pic_size):
        self.path_name = path_name
        self.pics_name = pics_name
        self.pic_size = pic_size
        self.num_pics = None
        self.reshaped_pics = None
        self.mean_pic = None
        self.centered_pics = None
        self.svd = None
        self.subspace_pics = None
        self.eps_0 = None
        self.eps_1 = None
        
        
    # пути всех файлов с именем 00000x.jpg
    # также тут устанавливаем их кол-во
    
    def get_pics_paths(self):
        path_list = []
        
        for path, subdirs, files in os.walk(self.path_name):
            for name in files:
                if name.endswith(self.pics_name):
                    path_list.append(os.path.join(path, name))
                    
        self.num_pics = len(path_list)
        
        return path_list
    
    # берем все пикчи с именем 00000x.jpg, ресайзим их до 50*50, и получившиеся 50*50*3 представления кладем в 
    # массив 
    
    def get_pics(self):
        path_list = self.get_pics_paths()
        pics_list = []
        
        for path in path_list:
            pic = Image.open(path)
            # ресайз кстати можно
            pic = pic.resize((self.pic_size, self.pic_size))
            pics_list.append(np.array(pic))
            pic.close()
            
        
        return np.array(pics_list)
        
    # выводим случайно выбранный пик из всех 00000x.jpg (ресайзнув предварительно)
    
    def get_random_pic(self, seed=1):
        path_list  = self.get_pics_paths()
        random_pic = random.Random(seed).choice(path_list)
        pic = Image.open(random_pic)
        pic = pic.resize((self.pic_size, self.pic_size))
        pic = np.array(pic)
        print(random_pic)
        plt.imshow(pic)
        
    # берем все пикчи 00000x.jpg, превращаем их в вектора и кладем их в одну матрицу S как столбцы
    
    def get_reshaped_pics(self):
        
        if self.reshaped_pics is not None:
            return self.reshaped_pics
        
        all_pics = self.get_pics()
        pics_reshaped = np.zeros((self.pic_size ** 2, len(all_pics), 3))
        
        for i, pic in enumerate(all_pics):
            for j in range(3):
                pics_reshaped[:, i, j] = pic[:,:,j].reshape(-1,1)[:,0]
        
        self.reshaped_pics = pics_reshaped
        
        return pics_reshaped


    # считаем средний столбец в матрице S
    # + есть опция вывода: берем этот средний столбец, возвращаем его в вид 2d массива и выводим что получилось
    
    def get_mean_pic(self, show=False):
        
        if self.mean_pic is not None:
            return self.mean_pic
        
        all_pics_reshaped = self.get_reshaped_pics() if self.reshaped_pics is None else self.reshaped_pics
        mean_pic = np.mean(all_pics_reshaped, axis=1) # ДОБАВИТЬ ОПЦИЮ МЕДИАНЫ
        
        if show:  
            plt.imshow(mean_pic.reshape((self.pic_size, self.pic_size, 3)) / 256)
            
        self.mean_pic = mean_pic
        
        return mean_pic
    
    # из каждого столбца матрицы S вычитаем средний столбец, получаем новую матрицу А
    
    def get_centered_pics(self):
        
        if self.centered_pics is not None:
            return self.centered_pics
        
        pics_reshaped = deepcopy(self.get_reshaped_pics()) if self.reshaped_pics is None else deepcopy(self.reshaped_pics)
        mean_pic = self.get_mean_pic() if self.mean_pic is None else self.mean_pic
        
        for i in range(self.num_pics):
            pics_reshaped[:,i,:] -= mean_pic
        
        self.centered_pics = pics_reshaped
        
        return pics_reshaped
        
    # берем матрицу А. Превращаем ее в 3 матрицы по RGB и каждые 3 матрицы СВД раскладываем
    
    def compute_svd(self):
        
        if self.svd is not None:
            return self.svd
        
        centered_pics = self.get_centered_pics() if self.centered_pics is None else self.centered_pics
        ch_r = centered_pics[:,:,0]
        ch_g = centered_pics[:,:,1]
        ch_b = centered_pics[:,:,2]
        
        svd_r, svd_g, svd_b = np.linalg.svd(ch_r), np.linalg.svd(ch_g), np.linalg.svd(ch_b)
        
        self.svd = (svd_r, svd_g, svd_b)
        
        return svd_r, svd_g, svd_b
    
    # Подсчет скалярных проекций на базовые лица для каждого вектор-лица
    
    def get_subspace_pics(self):
        
        if self.subspace_pics is not None:
            return self.subspace_pics
        
        pics_reshaped = self.get_reshaped_pics() if self.reshaped_pics is None else self.reshaped_pics
        mean_pic = self.get_mean_pic() if self.mean_pic is None else self.mean_pic
        centered_pics = self.get_centered_pics() if self.centered_pics is None else self.centered_pics
        
        dim = np.min((np.linalg.matrix_rank(centered_pics[:,:,0]), 
                     np.linalg.matrix_rank(centered_pics[:,:,1]),
                     np.linalg.matrix_rank(centered_pics[:,:,2])))
        
        svd_r, svd_g, svd_b = self.compute_svd() if self.svd is None else self.svd
        
        result = np.zeros((dim, self.num_pics, 3))
        
        for i in pb(range(self.num_pics)):
            result[:,i,0] = svd_r[0][:,:dim].T @ (pics_reshaped[:,i,0] - mean_pic[:,0])
            result[:,i,1] = svd_g[0][:,:dim].T @ (pics_reshaped[:,i,1] - mean_pic[:,1])
            result[:,i,2] = svd_b[0][:,:dim].T @ (pics_reshaped[:,i,2] - mean_pic[:,2])
        
        self.subspace_pics = result
        
        return result
    
    def calc_distances(self, pic_new):
        
        projections = self.get_subspace_pics() if self.subspace_pics is None else self.subspace_pics
        r, g, b = self.compute_svd() if self.svd is None else self.svd
        mean_pic = self.get_mean_pic() if self.mean_pic is None else self.mean_pic
        
        dim = projections.shape[0]
        
        pic_projection_r = r[0][:,:dim].T @ (pic_new[:,0] - mean_pic[:,0])
        pic_projection_g = g[0][:,:dim].T @ (pic_new[:,1] - mean_pic[:,1])
        pic_projection_b = b[0][:,:dim].T @ (pic_new[:,2] - mean_pic[:,2])
        
        distances = []
        
        for i in range(projections.shape[1]):
    
            e_r = np.linalg.norm(projections[:,i,0] - pic_projection_r)
            e_g = np.linalg.norm(projections[:,i,1] - pic_projection_g)
            e_b = np.linalg.norm(projections[:,i,2] - pic_projection_b)
    
            dist = np.mean([e_r, e_g, e_b])
            distances.append((i, dist))

        return distances 
    
    def compute_eps_1(self, path_to_nofaces=None):
        
        if self.eps_1 is not None:
            return self.eps_1
        
        path_list = []
        
        for path, subdirs, files in os.walk(path_to_nofaces):
            for name in files:
                if name.endswith('.jpg'):
                    path_list.append(os.path.join(path, name))
                    
        pics_list = []
        
        for path in path_list:
            pic = Image.open(path)
            # ресайз кстати можно
            pic = pic.resize((self.pic_size, self.pic_size))
            pics_list.append(np.array(pic))
            pic.close()
            
        pics_list = np.array(pics_list)
        pics_reshaped = np.zeros((self.pic_size ** 2, len(pics_list), 3))
        
        for i, pic in enumerate(pics_list):
            for j in range(3):
                pics_reshaped[:, i, j] = pic[:,:,j].reshape(-1,1)[:,0]
        
        eps_1_vals = []
                    
        projections = self.get_subspace_pics() if self.subspace_pics is None else self.subspace_pics
        r, g, b = self.compute_svd() if self.svd is None else self.svd
        mean_pic = self.get_mean_pic() if self.mean_pic is None else self.mean_pic
        dim = projections.shape[0]
        
        for i in pb(range(len(pics_list))):

            pic_projection_r = r[0][:,:dim].T @ (pics_reshaped[:,i,:][:,0] - mean_pic[:,0])
            pic_projection_g = g[0][:,:dim].T @ (pics_reshaped[:,i,:][:,1] - mean_pic[:,1])
            pic_projection_b = b[0][:,:dim].T @ (pics_reshaped[:,i,:][:,2] - mean_pic[:,2])

            f_p_r = r[0][:,:dim] @ pic_projection_r
            f_p_g = g[0][:,:dim] @ pic_projection_g
            f_p_b = b[0][:,:dim] @ pic_projection_b

            e_f_r = np.linalg.norm((pics_reshaped[:,i,:][:,0] - mean_pic[:,0]) - f_p_r)
            e_f_g = np.linalg.norm((pics_reshaped[:,i,:][:,1] - mean_pic[:,1]) - f_p_g)
            e_f_b = np.linalg.norm((pics_reshaped[:,i,:][:,2] - mean_pic[:,2]) - f_p_b)

            eps_1_vals.append(np.mean([e_f_r, e_f_g, e_f_b]))
            
        eps_1 = np.mean(eps_1_vals)
        
        self.eps_1 = eps_1
        
        return eps_1
        
    def compute_eps_0(self):
        
        if self.eps_0 is not None:
            return self.eps_0
        
        reshaped_pics = self.get_reshaped_pics()
        second_dists = []
        
        for i in pb(range(self.num_pics)):
            cur_dists = self.calc_distances(reshaped_pics[:,i,:])
            cur_dists = sorted(cur_dists, key=lambda tup: tup[1])
            second_dists.append(cur_dists[1][1])
            
        eps_0 = np.mean(second_dists)
        self.eps_0 = eps_0
        
        return eps_0
    
    def recognize(self, pic, show=True, path_to_nofaces=None):
        
        eps_0 = self.compute_eps_0() if self.eps_0 is None else self.eps_0
        eps_1 = self.compute_eps_1(path_to_nofaces) if self.eps_1 is None else self.eps_1
        
        projections = self.get_subspace_pics() if self.subspace_pics is None else self.subspace_pics
        r, g, b = self.compute_svd() if self.svd is None else self.svd
        mean_pic = self.get_mean_pic() if self.mean_pic is None else self.mean_pic
        dim = projections.shape[0]
        
        pic_r = pic[:,:,0].reshape(-1,1)[:,0]
        pic_g = pic[:,:,1].reshape(-1,1)[:,0]
        pic_b = pic[:,:,2].reshape(-1,1)[:,0]

        pic_vector = np.stack((pic_r, pic_g, pic_b), axis=1)
        
        pic_projection_r = r[0][:,:dim].T @ (pic_vector[:,0] - mean_pic[:,0])
        pic_projection_g = g[0][:,:dim].T @ (pic_vector[:,1] - mean_pic[:,1])
        pic_projection_b = b[0][:,:dim].T @ (pic_vector[:,2] - mean_pic[:,2])

        f_p_r = r[0][:,:dim] @ pic_projection_r
        f_p_g = g[0][:,:dim] @ pic_projection_g
        f_p_b = b[0][:,:dim] @ pic_projection_b

        e_f_r = np.linalg.norm((pic_vector[:,0] - mean_pic[:,0]) - f_p_r)
        e_f_g = np.linalg.norm((pic_vector[:,1] - mean_pic[:,1]) - f_p_g)
        e_f_b = np.linalg.norm((pic_vector[:,2] - mean_pic[:,2]) - f_p_b)
        
        eps_f = np.mean([e_f_r, e_f_g, e_f_b])
        
        if eps_f > eps_1:
            print('Not a face!')
            return False
        
        dists = self.calc_distances(pic_vector)
        dists_val = np.array([val[1] for val in dists])
        
        if len(dists_val[dists_val < eps_0]) == 0:
            print('Unknown face!')
            return False
        
        dists = sorted(dists, key=lambda tup: tup[1])
        
        pred_pic_id = dists[0][0]
        
        if show:
            path_list = self.get_pics_paths()
            pred_path = path_list[pred_pic_id]
            pic = Image.open(pred_path)
            pic = pic.resize((self.pic_size, self.pic_size))
            pic = np.array(pic)
            print(pred_path)
            plt.imshow(pic)
            
        return pred_pic_id
        
        
