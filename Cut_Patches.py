
def ObtPatches(img, punto, size_patch):
     
     cut_u_1 = int(punto[0] - size_patch[1]/2)
     cut_u_2 = int(punto[0] + size_patch[1]/2)
     cut_v_1 = int(punto[1] - size_patch[0]/2)
     cut_v_2 = int(punto[1] + size_patch[0]/2)

     img = img[cut_v_1:cut_v_2, cut_u_1:cut_u_2]
     
     return img