const tf = require('@tensorflow/tfjs');

export class Action {
    x = 0
    y = 0
    nx = 0
    ny = 0
    res= ''
    props = {}
    step = 0
    // f = []

    constructor(props: any) {
        this.props = props
        this.setProp(props)
    }

    setProp(props: any) {
        Object.keys(props).forEach((key: string) => {
            const self : any =  this
            if (self[key]) {
                self[key] = props[key]
            }
        })
    }

    get q() {
        let _q = 0
        const _zeta = this.zeta
        const condition : any = {
            1: _zeta >= 0 && _zeta <= 90,
            2: _zeta > 90 && _zeta <= 180,
            3: _zeta > 180 && _zeta <= 270,
            4: _zeta > 270 && _zeta <= 360,
        }

        Object.keys(condition).forEach((key : number | string) => {
            if (condition[key]) {
                _q = Number(key)
            }
        })
        // console.log(_q, _zeta, condition[_q] )
        return _q
    }

    get diff_x () { return this.nx - this.x }
    get diff_y () { return this.ny - this.y }

    get dist() {
        return (Math.sqrt(Math.pow(this.x - this.nx, 2) + Math.pow(this.y - this.ny, 2)))
    }

    get zeta() {
        const y = (this.y - this.ny)
        const x =  (this.x - this.nx)
        if (y == 0 || y == 0) {
            return 0
        }
        return Math.atan2(y, x) * 180 / Math.PI;
    }

    get slope() {
        return this.diff_y == 0 || this.diff_x == 0 ? 0 : this.diff_y/this.diff_x
    }

    get tensor() {
        const g = (value: number) => {
            if (value == 0) { return 0 }
            else if (value > 0) { return 1  }
            else if (value < 0) { return -1 }
        }

        const data  = [
            g(this.diff_x),
            g(this.diff_y),
            Math.ceil(this.zeta),
        ]

        console.log('mark', data)
       
        return tf.tensor(data)
    }
    toTensor_step() {
        const l = (value: number) => {
            value = Math.abs(value)
            // if (value <= 40) { return value }
            if (value <= 1) { return 'L0' }
            else if (value <= 2.5) { return 'L0.25' }
            else if (value <= 5) { return 'L0.5' }
            else if (value <= 10) { return 'L1' }
            else if (value <= 20) { return 'L2' }
            else if (value <= 25) { return 'L2.5' }
            else if (value <= 30) { return 'L3' }
            else if (value <= 50) { return 'L3' }
            else if (value <= 70) { return 'L7' }
            else if (value <= 110) { return 'L110' }
            else if (value <= 130) { return 'L130' }
            else { return 'OUT'  }
        }

        const data = [
            parseFloat(this.dist.toFixed(2)),
            this.res,
            l(this.dist),
        ]
        // console.log('walk-step toTensor_step', data)
        return tf.tensor(data)
    }
}