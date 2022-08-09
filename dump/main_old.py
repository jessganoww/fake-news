# train_headline_embedding, train_body_embedding, df_train = transform_to_matrix(df_train_headline, df_train_body,
#                                                                                'train')
# test_headline_embedding, test_body_embedding, df_test = transform_to_matrix(df_test_headline, df_test_body, 'test')
#
# df_train, df_val = split_train_val(df_train)
#
#
# print('Creating dataset...')
# train_ds = EmbeddingTrainDataset(df_train, train_headline_embedding, train_body_embedding)
# val_ds = EmbeddingTrainDataset(df_val, train_headline_embedding, train_body_embedding)
# test_ds = EmbeddingTestDataset(df_test, test_headline_embedding, test_body_embedding)
#
# class_weights = 1. / np.unique(df_train['Stance'], return_counts=True)[1]
# samples_weights = torch.from_numpy(np.array([class_weights[s] for s in df_train['Stance']]))
# sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), len(samples_weights))
#
# train_dl = DataLoader(train_ds, batch_size=16, sampler=sampler)
# val_dl = DataLoader(val_ds, batch_size=16)
# test_dl = DataLoader(test_ds, batch_size=16)
#
# curr_model = models_dict.get(MODEL_ID)
# optim_adam = torch.optim.Adam(curr_model.parameters()) # experiment with learning rate
# loss_CE = nn.CrossEntropyLoss()
#
#
# train_accuracy_list, train_loss_list, val_accuracy_list, val_loss_list = train_model(curr_model, optim_adam, loss_CE,
#                                                                                      EPOCHS, train_dl, val_dl,
#                                                                                      MODEL_ID)
#
# predictions = test_model(MODEL_ID, test_dl)
# predictions = [*map(predictions_dict.get, predictions)]
#
# test_stance = pd.read_csv("data/competition_test_stances.csv")['Stance']
# score, cm = score_submission(test_stance, predictions)
# test_accuracy = (np.unique(test_stance == predictions, return_counts=True)[1][1]) / len(predictions)
# print_confusion_matrix(cm)

# accuracy = train_model(model_B, optim_adam, loss_CE, EPOCHS, train_dl, 'Net3')
# predictions = test_model(model_B, loss_CE, test_dl, 'Net3')
# predictions = [*map(predictions_dict.get, predictions)]
#
# old_ids = df_test_body['Body ID']
# test_pred_ids = [old_ids[i] for i in list(df_test_headline['Body ID'])]
# test_pred = pd.DataFrame(data={'Headline': df_test_headline['Headline'], 'Body ID': test_pred_ids , 'Stance': predictions})
# test_pred.to_csv('answer.csv', index=False, encoding='utf-8')

