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
        colors.map((color, index) => [color, index])
    )
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )

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
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ])

    // Normaliza idade média do produto e aplica peso (0.1)
    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
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

    // Concatena todas as features em um único vetor
    return tf.concat1d([price, age, category, color])
}

// Treina o modelo com dados dos usuários
// Calcula vetores para todos os produtos
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

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

// Função para gerar recomendações para um usuário (implementar)
function recommend({ user }) {
    // TODO: Implementar lógica de recomendação usando vetores
}

// Mapeia ações para handlers correspondentes
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

// Listener para mensagens do thread principal
self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};