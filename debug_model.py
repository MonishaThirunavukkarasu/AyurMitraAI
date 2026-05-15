from app import load_artifacts, preprocess_user_input, model, label_encoder

load_artifacts()

print('Model loaded:', model is not None)
print('Label classes:', len(label_encoder.classes_))
print('Sample classes:', label_encoder.classes_[:5])

X_struct, X_text = preprocess_user_input('joint pain and fatigue', 35, 75, 6, 5)
if X_struct is not None and X_text is not None:
    print('X_struct', X_struct.shape, 'X_text', X_text.shape)

    preds = model.predict([X_struct, X_text], verbose=0)[0]
    print('preds len', len(preds), 'max', preds.max())

    best = preds.argmax()
    print('best idx', best, 'class', label_encoder.inverse_transform([best])[0], 'prob', preds[best])

    for i in (-preds).argsort()[:5]:
        print('top', i, label_encoder.inverse_transform([i])[0], preds[i])
else:
    print('Preprocessing failed')
