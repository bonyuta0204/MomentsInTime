
for data in DataLoader:
    data.build_image_from_movie(save=True)
    sum_activation = 0;
    for image in data.cropped_images():
        sum_activation += net(image)
    # something to get label in text or index
    label = hogehoge
    data.label = label

