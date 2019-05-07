import axios from 'axios'
import { Message } from 'element-ui'
import qs from 'qs'
import { toCamelCase } from './tools'

const ajax = {
  async get (url, data) {
    try {
      let res = await axios.get(url, {params: data})
      return new Promise((resolve) => { resolve(toCamelCase(res.data)) })
    } catch (err) {
      this.handleResponseError()
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
      const res = await axios.post(url, opt.headers && opt.headers['Content-Type'].indexOf('application/json') > -1 ? data : qs.stringify(data), config)
      return new Promise((resolve, reject) => { resolve(toCamelCase(res.data)) })
    } catch (err) {
      this.handleResponseError(err)
    }
  },
  async delete (url) {
    try {
      let res = await axios.delete(url)
      return new Promise((resolve) => { resolve(toCamelCase(res.data)) })
    } catch (err) {
      this.handleResponseError()
    }
  },

  handleResponseError(err) {
    const errors = {
      500: 'Server Internal Error',
      422: 'Submit data is invalid',
      409: 'Record already exists'
    }
    Message.closeAll()
    Message.error(errors[err.response.status] || 'Unknown Error')
  }
}

export {
  ajax
}
