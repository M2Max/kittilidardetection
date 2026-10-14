def classify(box):
    l, w, h = box[3], box[4], box[5]                        # l >= w sempre (lo garantisce box_bev)
    if h < 1.0 or h > 2.5 or l > 6.0 or w > 2.5:            # fuori misura: cordoli, rumore, muri, alberi
        return None
    if w > 1.0 or l > 2.2:                                  # largo o lungo come un'auto (anche se vista a metà)
        return "Car"
    if h < 1.2:                                             # stretto e basso: cespuglio, cassonetto, paletto
        return None
    if l > 1.2:                                             # stretto ma lungo come una bici
        return "Cyclist"
    return "Pedestrian"                                     # stretto e corto: una persona
