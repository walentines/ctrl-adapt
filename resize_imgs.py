from PIL import Image

def resize_image(input_path, output_path, size=(768, 512)):
    # Open the input image
    image = Image.open(input_path)
    
    # Resize the image
    resized_image = image.resize(size)
    
    # Save the resized image
    resized_image.save(output_path)
    print(f"Image saved to {output_path} with size {resized_image.size}")

# Example usage
if __name__ == "__main__":
    root_path = "/bigdata/userhome/ionut.serban/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full/"
    location = "munster"
    # location = "frankfurt"
    location = "lindau"

    path_gt = root_path + "leftImg8bit/val/" + location + "/"
    path_depth = root_path + "crestereo_depth/val/" + location + "/"
    path_segm = root_path + "gtFine/val/" + location + "/"
    image_name =  "lindau_000012_000019"
    input_file_gt = path_gt + image_name + "_leftImg8bit.png"
    input_file_segm = path_segm + image_name + "_gtFine_color.png"
    input_file_depth = path_depth + image_name + "_crestereo_depth.png"

    output_folder = "/bigdata/userhome/ionut.serban/shared/MIRPR-proiectCONTROL-NET/TrainingControlnetNew/new_imgs/" 
    output_file_gt = output_folder + image_name + "_ground_truth.png"
    output_file_segm = output_folder + image_name + "_segmentation.png"
    output_file_depth = output_folder + image_name + "_depth.png"
    resize_image(input_file_gt, output_file_gt)
    resize_image(input_file_segm, output_file_segm)
    resize_image(input_file_depth, output_file_depth)

