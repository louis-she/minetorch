<template>
  <div class="m-term">
    <div
      ref="term"
      class="m-term-client"></div>
  </div>
</template>

<script>
import { Terminal } from 'xterm'
import io from 'socket.io-client'
import * as fit from 'xterm/lib/addons/fit/fit'

export default {
  props: {
    rows: {
      type: Array,
      default: () => []
    },
    // socket地址
    attach: {
      type: String,
      default: () => ''
    }
  },
  data () {
    return {
      term: null,
      socket: null
    }
  },
  watch: {
    rows (val) {
      val.forEach(v => {
        this.term.writeln(v)
      })
    }
  },
  mounted () {
    Terminal.applyAddon(fit)
    this.term = new Terminal({
      theme: {
        background: '#212121'
      }
    })

    if (this.attach) {
      this.socket = io(this.attach)

      this.socket.on('connect', () => {
        console.log('client connected')
      })

      this.socket.on('new_server_log', (data) => {
        this.term.write(data)
        data.split('\n').forEach((line) => {
          this.term.writeln(line)
        })
      })

      this.socket.on('disconnect', () => {
        console.log('client disconnected')
      })
    }
  },
  methods: {
    open() {
      this.term.open(this.$refs.term)
    }
  }
}
</script>

<style src="xterm/dist/xterm.css"></style>
<style lang="scss">
.m-term {
  .xterm {
    padding: 8px 10px;
  }
}
</style>
