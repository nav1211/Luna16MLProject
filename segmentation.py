from skimage import morphology, measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import numpy as np

working_path = ".\\dataset\\volumes_modified\\"
file_list = glob(working_path+"images_*.npy")

for image_file in file_list:
    processingImages = np.load(image_file).astype(np.float16)
    print("on image", image_file)
    for i in range(len(processingImages)):
        image = processingImages[i]
        std = np.std(image)
        mean = np.mean(image)
        image = image-mean
        image = image/std
        middle = image[100:400,100:400]
        mean = np.mean(middle)
        min = np.min(image)
        max = np.max(image)
        image[image == max] = mean
        image[image == min] = mean
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_image = np.where(image<threshold,1.0,0.0)
        eroded = morphology.erosion(thresh_image,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)
        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        processingImages[i] = mask
    np.save(image_file.replace("images", "lungmask"),processingImages)

file_list=glob(working_path+"lungmask_*.npy")
out_images = []      #final set of images
out_nodemasks = []   #final set of nodemasks
for fname in file_list:
    print ("working on file ", fname)
    processingImages = np.load(fname.replace("lungmask", "images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask", "masks"))
    for i in range(len(processingImages)):
        mask = masks[i]
        node_mask = node_masks[i]
        image = processingImages[i]
        new_size = [512,512]
        image= mask*image
        new_mean = np.mean(image[mask > 0])
        new_std = np.std(image[mask > 0])
        old_min = np.min(image)
        image[image == old_min] = new_mean-1.2*new_std
        image = image - new_mean
        image = image / new_std
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        min_row = 512
        min_col = 512
        max_row = 0
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        height = max_row - min_row
        width = max_col - min_col
        height = max_row - min_row
        if width > height:
            max_row=min_row+width
        else:
            max_col = min_col+height
        image = image[min_row:max_row,min_col:max_col]
        mask = mask[min_row:max_row,min_col:max_col]
        if max_row-min_row < 5 or max_col-min_col< 5:
            pass
        else:
            mean = np.mean(image)
            image = image - mean
            min = np.min(image)
            max = np.max(image)
            image = image/(max-min)
            new_image = resize(image, [512,512])
            new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
            out_images.append(new_image)
            out_nodemasks.append(new_node_mask)

num_images = len(out_images)
final_images = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
final_masks = np.ndarray([num_images, 1, 512, 512], dtype=np.float32)
for i in range(num_images):
    final_images[i, 0] = out_images[i]
    final_masks[i, 0] = out_nodemasks[i]

rand_i = np.random.choice(range(num_images), size=num_images, replace=False)
test_i = int(0.2 * num_images)
np.save(working_path+"trainImages.npy", final_images[rand_i[test_i:]])
np.save(working_path+"trainMasks.npy", final_masks[rand_i[test_i:]])
np.save(working_path+"testImages.npy", final_images[rand_i[:test_i]])
np.save(working_path+"testMasks.npy", final_masks[rand_i[:test_i]])
