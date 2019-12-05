

def use_predicted_probability(label_dict, pred_key):
    return label_dict[pred_key]


def convert_card_list(card_list_input):
    out_array = []
    for card in card_list_input:
        if card== 'buster':
            out_array.append(1)

        if card== 'arts':
            out_array.append(2)
        if card== 'quick':
            out_array.append(3)
    #print('outarray: ',out_array)
    #out_array2 = np.reshape(np.asarray(out_array), (1, 5))
    return out_array