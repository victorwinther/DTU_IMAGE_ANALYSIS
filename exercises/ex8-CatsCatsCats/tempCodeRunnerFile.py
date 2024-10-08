 w = 1
    synth_cat = mean_cat_vector + w * cats_pca.components_[0, :]
    synth_cat_image = create_u_byte_image_from_vector(synth_cat, height, width, channels)
    plt.imshow(synth_cat_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    