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

        return tf.tensor(data)
    }
}