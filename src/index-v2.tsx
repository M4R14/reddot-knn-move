import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import { makeid } from './function/makeid'

class Pos {
    protected id: string;
    public x = 0
    public y = 0
    public r = 2
    public color = "#fff"

    constructor() {
        this.id = makeid(8)
    }

    render(context: CanvasRenderingContext2D) {
        context.beginPath();
        context.fillStyle = this.color;
        context.arc(this.x, this.y, this.r, 0, 2 * Math.PI);
        context.fill();
        context.closePath();
    }
}

class Food extends Pos {}
class Dot extends Pos {
    ml: boolean = false
    zeta = 0
    speed = 2
    constructor() {
        super()
        this.r = 4
        this.color = 'red'
    }

    walk() {
        this.x += Math.cos(this.zeta) * this.speed
        this.y += Math.sin(this.zeta) * this.speed
    }
}

const craeteFood = (num_of_dot: number, width: number, height: number) => {
    let _data = []
    for (let index = 0; index < num_of_dot; index++) {
        const data = new Food();
        data.x = parseInt((Math.random() * (width )).toString())
        data.y = parseInt((Math.random() * (height )).toString())
        _data.push(data)
    }

    return _data
}

const craeteDot = (num_of_dot: number, width: number, height: number) => {
    let _dots = []
    for (let index = 0; index < num_of_dot; index++) {
        const dot = new Dot();
        dot.x = parseInt((Math.random() * width).toString())
        dot.y = height / 2
        _dots.push(dot)
    }

    return _dots
}



var canvas: HTMLCanvasElement, app, ml: HTMLInputElement
window.onload = async () => {
    ml = document.createElement("input");
    ml.type = 'checkbox'
    app = document.getElementById("app");
    canvas = document.createElement("canvas");
    canvas.width = window.innerWidth - window.innerWidth/20
    canvas.height = window.innerHeight - window.innerHeight/20
    canvas.style.backgroundColor = 'black'
    app.appendChild(ml);
    app.appendChild(document.createTextNode("หยุดการเรียนรู้"));
    app.appendChild(document.createElement("br"));
    app.appendChild(canvas);

    const classifier = knnClassifier.create();
    await classifier.addExample(tf.tensor([1, 1]), 45);

    const dots: Dot[]  = craeteDot(6, canvas.width, canvas.height)
    dots[0].ml = true
    dots[1].ml = true
    dots[2].ml = true
    let foods: Food[]  = craeteFood(200, canvas.width, canvas.height)

    const render = () => {
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, canvas.width, canvas.height);

        // console.log({dots})
        dots.forEach(async (d: Dot) => {
            d.render(context);
            const _x = d.x;
            const _y = d.y;
            const target: {
                food: Food;
                dist: number;
            } = foods.reduce((f0, f1: Food) => {
                const dist_f1 = Math.sqrt(Math.pow(f1.x - _x, 2) + Math.pow(f1.y - _y, 2));
                if (f0.dist > dist_f1) {
                    return { food: f1, dist: dist_f1 };
                }
                else {
                    return f0;
                }
            }, { food: new Food(), dist: Infinity });

            context.beginPath();
            context.strokeStyle = d.color;
            context.moveTo(d.x, d.y);
            context.lineTo(target.food.x, target.food.y);
            context.stroke();
            context.closePath();
            
            let _zeta = [ 0, 90, 180, 270 ][ Math.ceil(Math.random() * 3) ];
            let predict: any = null;
            const tensor = tf.tensor([target.food.x - _x, target.food.y - _y]);
            if (d.ml) {
                predict = await classifier.predictClass(tensor, 3);
                _zeta = Number(predict.label);
            }
            d.zeta = _zeta;
            d.speed = 2
            if (d.ml && predict == null) {
                d.speed = 0
            }
           
            d.walk();
            
            const dist_aft = Math.sqrt(Math.pow(target.food.x - d.x, 2) + Math.pow(target.food.y - d.y, 2));
            if (dist_aft < target.dist) {
                d.color = "green";
                if (d.ml) {
                    d.color = "blue";
                } 
                if (ml.checked == false) {
                    await classifier.addExample(tensor, _zeta);
                }
            } else {
                d.color = "red";
            }
            // if (d.ml && predict != null) {
            //     console.log('####', d.color, predict);
            // }

            context.beginPath();
            context.fillStyle = d.color;
            context.fillText(`zeta: ${d.zeta}`, d.x + d.r + 2, d.y + 2);
            context.closePath();

            
        }) 
        
        foods = foods.filter(d => {
            let survival = true
            dots.forEach(dt => {
                if (dt.x + dt.r >= d.x && dt.x <= d.x + d.r) {
                    if (dt.y + dt.r >= d.y && dt.y <= d.y + d.r) {
                        survival = false
                    }
                }
            })
            return survival
        })

        foods.forEach(d => {
            d.render(context)
        })   

        requestAnimationFrame(render)     
    }

    render()
}