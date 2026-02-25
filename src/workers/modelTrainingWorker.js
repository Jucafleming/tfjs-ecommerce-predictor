import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

// Armazena o contexto global com dados dos produtos e usuários
let _globalCtx = {};
// Será usado para armazenar o modelo treinado no futuro
let _model = null

// Pesos para balancear a importância de cada feature na recomendação
// Categoria é mais importante (0.4), cor é menos importante (0.1)
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};

// Converte valores contínuos (preço, idade) para escala 0–1
// Necessário para equilibrar features diferentes na rede neural
// Fórmula: (valor - mín) / (máx - mín)
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

// Prepara os dados dos usuários e produtos para treinamento
// Calcula índices para cores/categorias e normaliza valores
function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    // Encontra min/max para normalização posterior
    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    // Cria mapeamento: cor/categoria → índice numérico
    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // Calcula a idade média dos compradores por produto
    // Ajuda a criar padrões: produtos para jovens vs. idosos
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    // Normaliza a idade média de cada produto para 0–1
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // Dimensão total do vetor: preço(1) + idade(1) + categorias + cores
        dimentions: 2 + categories.length + colors.length
    }
}

// Cria one-hot encoding ponderado pela importância da feature
// Ex: categoria 2 em 5 categorias → [0, 0, 0.4, 0, 0]
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

// Transforma um produto em vetor numérico para a rede neural
// Combina: preço, idade média, categoria e cor em um único vetor
function encodeProduct(product, context) {
    // Normaliza preço para 0–1 e aplica peso (0.2)
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    // Normaliza idade média do produto e aplica peso (0.1)
    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    // One-hot encoding da categoria com peso (0.4)
    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    // One-hot encoding da cor com peso (0.3)
    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 :
                        0
                )
                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)

            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
// Exemplo: [preço_normalizado, idade_normalizada, cat_one_hot..., cor_one_hot...]
//
// Suponha categorias = ['acessórios', 'eletrônicos', 'vestuário']
// Suponha cores      = ['preto', 'cinza', 'azul']
//
// Para Rafael (idade 27, categoria: acessórios, cores: preto/cinza),
// o vetor poderia ficar assim:
//
// [
//   0.45,            // peso do preço normalizado
//   0.60,            // idade normalizada
//   1, 0, 0,         // one-hot de categoria (acessórios = ativo)
//   1, 0, 0          // one-hot de cores (preto e cinza ativos, azul inativo)
// ]
//
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    
    // Carrega dados dos produtos
    const products = await (await fetch('/data/products.json')).json()

    // Prepara contexto com dados normalizados
    const context = makeContext(products, users)
    
    // Converte cada produto em vetor numérico
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    })

    // Armazena contexto globalmente para usar em recomendações
    _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

// Função para gerar recomendações para um usuário (implementar)
function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    // Converte o usuário fornecido no vetor de features codificadas
    // (preço ignorado, idade normalizada, categorias ignoradas)
    // Isso transforma as informações do usuário no mesmo formato numérico
    // que foi usado para treinar o modelo.
    const userVector = encodeUser(user, context).dataSync()

    // Em aplicações reais:
    // Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    // Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    // Execute _model.predict() apenas nesses produtos

    // Crie pares de entrada: para cada produto, concatene o vetor do usuário
    // com o vetor codificado do produto.
    // Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto).
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // Converta todos esses pares (usuário, produto) em um único Tensor.
    // Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs)

   

    // Rode a rede neural treinada em todos os pares (usuário, produto) de uma vez.
    // O resultado é uma pontuação para cada produto entre 0 e 1.
    // Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(inputTensor)

    // Extraia as pontuações para um array JS normal.
    const scores = predictions.dataSync()
    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // previsão do modelo para este produto
        }
    })

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)

    // Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

// Listener para mensagens do thread principal
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};