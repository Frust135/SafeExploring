def create_csv(col_names, state_array, action_array, qvalue_array, danger_state_array):
    import csv
    file_name = 'data.csv'
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(col_names)

        for i in range(len(state_array)):
            row = [
                state_array[i],
                action_array[i],
                qvalue_array[i],
                danger_state_array[i],
            ]
            writer.writerow(row)
    print('CSV finalizado')
    return True