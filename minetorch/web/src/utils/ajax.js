import axios from 'axios'
import { Message } from 'element-ui'
import qs from 'qs'

const ajax = {
  async get (url, data) {
    try {
      let res = await axios.get(url, {params: data})
      res = res.data
      return new Promise((resolve) => {
        if (res.code === 0) {
          resolve(res)
        } else {
          Message.closeAll()
          Message.error(res.message)
          resolve(res)
        }
      })
    } catch (err) {
      Message.closeAll()
      Message.error('服务器出错')
      console.log(err)
    }
  },
  async post (url, data, opt = {}) {
    try {
      let config = Object.assign({
        timeout: 50000,
        responseType: 'json',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
        }
      }, opt)
      let res = await axios.post(url, opt.headers && opt.headers['Content-Type'].indexOf('application/json') > -1 ? data : qs.stringify(data), config)
      res = res.data
      return new Promise((resolve, reject) => {
        if (res.code === 0) {
          resolve(res)
        } else {
          Message.closeAll()
          Message.error(res.message)
          reject(res)
        }
      })
    } catch (err) {
      Message.closeAll()
      Message.error('服务器出错')
      console.log(err)
    }
  }
}

export {
  ajax
}
