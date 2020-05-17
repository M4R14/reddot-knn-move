import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import { Action } from "./class/Action";
import { People } from "./class/People";
import { Food } from "./class/Food";

declare global {
    interface ObjectConstructor {
      fromEntries(xs: [string|number|symbol, any][]): object
    }
}

let dataInit = [
    new Action({ y: 0, x: 0, res: 'x', step: 0 }),
];

const classifier_save = (_classifier: { getClassifierDataset: () => any; } | null, localStorageKey: string) => {
    // Save it to a string:
    let str = JSON.stringify(
        Object.entries(_classifier.getClassifierDataset())
            .map(([label, data] : [ any, any ])=>{
                return [
                     label,
                     Array.from(data.dataSync()),
                     data.shape
                ]
            })
    )

    localStorage.setItem(localStorageKey, str);
}

const classifier_load = (localStorageKey: string) => {
    let _classifier = knnClassifier.create();

    var str = localStorage.getItem(localStorageKey);
    if (str == null) {
        return _classifier;
    }
    // Load it back into a fresh classifier:
    _classifier.setClassifierDataset(
        Object.fromEntries(
            JSON.parse(str).map(([label, data, shape] : [ any, any, any ])=>{
                return [label, tf.tensor(data, shape)];
            })
        ) 
    );

    return _classifier;
}

export let eat_food = 0
export let classifier : any  = null;

const init = async function() {
    // Create the classifier.
    classifier = classifier_load('classifier-nav');
}

init();

const distance = (xa: number, ya: number, xb: number, yb: number ) => {
    return Math.sqrt(Math.pow(xa - xb, 2) + Math.pow(ya - yb, 2))
}

function update (nodes: People[], foods: Food[]) : void {
    nodes.forEach(async (node : People) => {
        node.friends = nodes.filter(nd => nd.id != node.id)
        await node.think_v2()
        if (node.step > 0) {
            const _foods = foods
            node.set_target(_foods)
            node.setColor()

            node.walk()
        }
        node.eat(foods, (next) => {foods = next})
    })
}

function createChain (canvas: HTMLCanvasElement, n: number, radius = 5) {
    const _nodes = [];
    
    for (let i = 0; i < n; i ++) {
      _nodes.push(new People(radius, canvas));
    }

    return _nodes;
}

const createFoods = function(canvas: HTMLCanvasElement, n: number) {
    const _foods = [];
    console.log('createFoods', canvas.width)
    for (let i = 0; i < n; i ++) {
        _foods.push(new Food(canvas.width, canvas.height));
    }

    return _foods;
}

window.onload = function() {
    // we'll put all our code within this function
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const canvas_place = document.getElementById('canvas-place');
    
    console.log('start')
    
    if (context == null || canvas_place == null) {
        return;
    }

    canvas_place.appendChild(canvas);
    let isMouseDown = false;
    canvas.width  = window.innerWidth;
    canvas.height = 500
    // styles -------------------------
    document.body.style.background = "#000000";
    canvas.style.background = "#152c40";
    canvas.style.display = "block";
    canvas.style.margin = "0 auto";
    // --------------------------------

    

    const draw = function() : void {
        context.clearRect(0, 0, canvas.width, canvas.height);

        nodes.forEach((node : People) => {
            context.beginPath();
            context.fillStyle = node.color;
            context.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);

            context.fillText(`score: ${node.score}`, node.x + node.radius + 2, node.y + 2);

            let confidences = 0
            if (node.prediction) {
                confidences = node.prediction.confidences[node.prediction.label]
                confidences = parseFloat(confidences.toFixed(4))
            }

            context.fillText(
                `${node.nav} (${confidences})` ,
                node.x + node.radius + 2,
                node.y - 8
            );

            context.arc(node.x, node.y, node.step, 0, 2 * Math.PI);

            if (node.target) {
                const dist = parseFloat(distance(node.x, node.y, node.target.x, node.target.y).toFixed(4))
                if (dist > 200) {
                    console.log({ "node.target": node.target })
                }
                context.fillText(
                    `dist: ${dist}` ,
                    node.x + node.radius + 2,
                    node.y + 12
                );

                context.moveTo(node.x, node.y);
                context.strokeStyle = node.color;
                context.lineTo(node.target.x, node.target.y);
                context.stroke();
            }

            context.fill();
            context.closePath();
        })

        foods.forEach(food => {
            context.beginPath();
            context.fillStyle = food.color;
            context.arc(food.x, food.y, food.radius, 0, 2 * Math.PI);
            context.fill();
            context.closePath();
        })
    };

    let count = 0
    const start : any  = new Date();
    const eLoop = document.getElementById('loop')|| { innerText: '' }
    const eDiffSec = document.getElementById('diff_sec') || { innerText: '' }
    
    const tick = async function() {
        const b : any = new Date();
        const diff_sec : any = (b - start) / 1000;
        eLoop.innerText = count.toString();
        eDiffSec.innerText = diff_sec.toString();

        update(nodes, foods)
        draw()

        count += 1
        if (count % 100 == 0) {
            if ('x' in classifier.getClassExampleCount()) {
                classifier.clearClass('x')
            }

            classifier_save(classifier, 'classifier-nav');
            console.log('classifier-nav', {
                getClassExampleCount:  classifier.getClassExampleCount(),
                getClassifierDataset:  classifier.getClassifierDataset(),
                getNumClasses:  classifier.getNumClasses(),
            })

            if (count % 500 == 0) {
                window.location.reload()
            }
        }
        requestAnimationFrame(tick);
    };


    canvas.onclick = function(e) {
        foods.push(new Food(e.offsetX, e.offsetY))
    };

    let foods = createFoods(canvas, 100);
    let nodes = createChain(canvas, 3); // you can also pass radius as a second param
    
    tick()

    const use_ai = document.getElementById('use-ai');
    if (use_ai) {
        use_ai .addEventListener('click', () => {
            nodes = nodes.map(n => {
                n.use_ai = !n.use_ai
                if (n.use_ai) {
                    n.main_color = 'green'
                } else {
                    n.main_color = "red"
                }
                return n
            })
        }) 
    }
    
}